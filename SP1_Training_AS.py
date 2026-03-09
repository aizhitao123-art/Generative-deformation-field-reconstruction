# -*- coding: utf-8 -*-
"""
DDPM training (no conditions) with Active Sampling.

Main features:
- Load deformation data from XLSX files
- Convert scattered X-Y-value samples into raster images
- Train a time-conditioned DDPM
- Support block-wise Active Sampling for efficient subset selection
- Save checkpoints and training logs

This script is written in English only and is intended for GitHub release.
"""

import os
import glob
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from torch.amp import autocast, GradScaler


# ============================================================
# 0) Utilities
# ============================================================
def seed_everything(seed: int = 42):
    print("Setting random seeds...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flush_epoch_log_to_xlsx(epoch_rows, xlsx_path, sheet_name="epoch_log"):
    if len(epoch_rows) == 0:
        return
    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
    df = pd.DataFrame(epoch_rows)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def save_checkpoint(path, ddpm, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model_state": ddpm.state_dict()}
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, ddpm, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if "model_state" in ckpt:
        ddpm.load_state_dict(ckpt["model_state"], strict=True)
    else:
        ddpm.load_state_dict(ckpt, strict=True)
    print(f"Loaded checkpoint: {path}")
    return ckpt


# ============================================================
# 1) Dataset: XLSX -> raster image tensor
# ============================================================
class XLSXImageDataset(Dataset):
    """
    Return image tensors with shape [1, H, W] normalized to [-1, 1].
    """

    def __init__(
        self,
        root_dir,
        image_size=(128, 128),
        deform_min=0.0,
        deform_max=1.0,
        polygon_points=None,
        sheet_name="Sheet1",
        skiprows=1,
        multiply_minus_one=False,
    ):
        print(f"Initializing dataset from: {root_dir}")

        self.image_size = image_size
        self.deform_min = float(deform_min)
        self.deform_max = float(deform_max)
        self.sheet_name = sheet_name
        self.skiprows = skiprows
        self.multiply_minus_one = bool(multiply_minus_one)

        self.polygon = Polygon(polygon_points) if polygon_points is not None else None
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.xlsx")))

        if len(self.files) == 0:
            raise RuntimeError(f"No .xlsx files found under: {root_dir}")

        from openpyxl import load_workbook

        def is_valid_xlsx(path: str) -> bool:
            try:
                wb = load_workbook(path, read_only=True, data_only=True)
                wb.close()
                return True
            except Exception:
                return False

        valid_files = []
        bad_files = []

        for fp in self.files:
            if is_valid_xlsx(fp):
                valid_files.append(fp)
            else:
                bad_files.append(fp)

        if len(bad_files) > 0:
            print(f"Found {len(bad_files)} invalid XLSX files. They will be skipped.")
            bad_list_path = os.path.join(root_dir, "_bad_xlsx_list.txt")
            with open(bad_list_path, "w", encoding="utf-8") as f:
                for b in bad_files:
                    f.write(b + "\n")
            print(f"Invalid file list saved to: {bad_list_path}")

        self.files = valid_files

        if len(self.files) == 0:
            raise RuntimeError(f"All XLSX files under {root_dir} are invalid.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        df_img = pd.read_excel(
            file_path,
            sheet_name=self.sheet_name,
            header=None,
            skiprows=self.skiprows,
            engine="openpyxl",
        )

        if df_img.shape[1] < 3:
            raise RuntimeError(
                f"Sheet '{self.sheet_name}' must contain at least 3 columns (X, Y, V): {file_path}"
            )

        x_coords = df_img.iloc[:, 0].astype(np.float32).values
        y_coords = df_img.iloc[:, 1].astype(np.float32).values
        values = df_img.iloc[:, 2].astype(np.float32).values

        if self.multiply_minus_one:
            values = -1.0 * values

        h, w = self.image_size

        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())

        x_img = np.linspace(x_min, x_max, w, dtype=np.float32)
        y_img = np.linspace(y_min, y_max, h, dtype=np.float32)
        xx, yy = np.meshgrid(x_img, y_img)
        pixel_centers = np.column_stack([xx.flatten(), yy.flatten()]).astype(np.float32)

        if self.polygon is not None:
            mask = np.array(
                [self.polygon.contains(Point(float(x), float(y))) for x, y in pixel_centers],
                dtype=bool,
            )
        else:
            mask = np.ones((pixel_centers.shape[0],), dtype=bool)

        tree = cKDTree(np.column_stack([x_coords, y_coords]).astype(np.float32))
        _, indices = tree.query(pixel_centers[mask], k=1)
        nearest_values = values[indices]

        denom = (self.deform_max - self.deform_min) + 1e-8
        values_norm = 2.0 * ((nearest_values - self.deform_min) / denom) - 1.0

        image_flat = np.zeros(h * w, dtype=np.float32)
        image_flat[mask] = values_norm
        image = image_flat.reshape(h, w)[None, ...]

        return torch.tensor(image, dtype=torch.float32)


# ============================================================
# 2) U-Net blocks
# ============================================================
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = (in_channels == out_channels)
        self.is_res = is_res

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            out = (x + x2) if self.same_channels else (x1 + x2)
            return out / 1.414
        return self.conv2(self.conv1(x))


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.block1 = ResidualConvBlock(out_channels + skip_channels, out_channels)
        self.block2 = ResidualConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            h = min(x.size(2), skip.size(2))
            w = min(x.size(3), skip.size(3))
            x = x[:, :, :h, :w]
            skip = skip[:, :, :h, :w]

        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_feat=256):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)
        self.down4 = UnetDown(4 * n_feat, 8 * n_feat)

        self.timeembed1 = EmbedFC(1, 8 * n_feat)
        self.timeembed2 = EmbedFC(1, 4 * n_feat)

        self.up1 = UnetUp(in_channels=8 * n_feat, skip_channels=4 * n_feat, out_channels=4 * n_feat)
        self.up2 = UnetUp(in_channels=4 * n_feat, skip_channels=2 * n_feat, out_channels=2 * n_feat)
        self.up3 = UnetUp(in_channels=2 * n_feat, skip_channels=1 * n_feat, out_channels=1 * n_feat)
        self.up4 = UnetUp(in_channels=1 * n_feat, skip_channels=1 * n_feat, out_channels=1 * n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        x0 = self.init_conv(x)
        d1 = self.down1(x0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        temb1 = self.timeembed1(t).view(-1, 8 * self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, 4 * self.n_feat, 1, 1)

        d4 = d4 + temb1
        u1 = self.up1(d4, d3)
        u1 = u1 + temb2
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x0)

        return self.out(u4)


# ============================================================
# 3) DDPM
# ============================================================
def ddpm_schedules(beta1, beta2, t_steps):
    beta_t = (beta2 - beta1) * torch.arange(0, t_steps + 1, dtype=torch.float32) / t_steps + beta1
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "oneover_sqrta": 1 / torch.sqrt(alpha_t),
        "sqrt_beta_t": torch.sqrt(beta_t),
        "alphabar_t": alphabar_t,
        "sqrtab": torch.sqrt(alphabar_t),
        "sqrtmab": torch.sqrt(1 - alphabar_t),
        "mab_over_sqrtmab": (1 - alpha_t) / torch.sqrt(1 - alphabar_t),
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super().__init__()
        print("Initializing DDPM...")
        self.nn_model = nn_model.to(device)
        self.n_T = int(n_T)
        self.device = device
        self.loss_fn = nn.MSELoss(reduction="none")

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

    def loss_per_sample(self, x, n_repeat=1):
        batch_size = x.shape[0]
        losses = []

        for _ in range(n_repeat):
            t_steps = torch.randint(1, self.n_T + 1, (batch_size,), device=self.device)
            noise = torch.randn_like(x)
            x_t = self.sqrtab[t_steps, None, None, None] * x + self.sqrtmab[t_steps, None, None, None] * noise
            t_norm = (t_steps / self.n_T).float().view(batch_size, 1)
            pred_noise = self.nn_model(x_t, t_norm)
            per_pixel = self.loss_fn(noise, pred_noise)
            per_sample = per_pixel.mean(dim=(1, 2, 3))
            losses.append(per_sample)

        return torch.stack(losses, dim=0).mean(dim=0)

    def forward(self, x):
        per_sample = self.loss_per_sample(x, n_repeat=1)
        return per_sample.mean()

    @torch.no_grad()
    def sample(self, n_sample, size, device):
        self.eval()
        x_i = torch.randn(n_sample, *size, device=device)

        for i in range(self.n_T, 0, -1):
            t_norm = torch.full((n_sample, 1), i / self.n_T, device=device)
            z = torch.randn_like(x_i) if i > 1 else 0
            eps = self.nn_model(x_i, t_norm)
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

        return x_i


# ============================================================
# 4) Active Sampling
# ============================================================
@torch.no_grad()
def compute_information_scores(ddpm, dataset, candidate_indices, config):
    ddpm.eval()

    sub_ds = Subset(dataset, candidate_indices)
    loader = DataLoader(
        sub_ds,
        batch_size=config["al_score_batch_size"],
        shuffle=False,
        num_workers=config.get("al_score_num_workers", 0),
        pin_memory=True,
        drop_last=False,
    )

    scores = np.zeros(len(candidate_indices), dtype=np.float32)
    ptr = 0

    for x in tqdm(loader, desc="Scoring active sampling candidates", leave=False):
        x = x.to(config["device"], non_blocking=True)
        with autocast("cuda"):
            per_sample_loss = ddpm.loss_per_sample(x, n_repeat=config["al_score_n_repeat"])

        b = per_sample_loss.shape[0]
        scores[ptr:ptr + b] = per_sample_loss.detach().float().cpu().numpy()
        ptr += b

    return scores


def select_active_indices(ddpm, dataset, epoch, config):
    total_n = len(dataset)

    if epoch < config["al_warmup_epochs"]:
        all_idx = list(range(total_n))
        random.shuffle(all_idx)
        k = config["al_subset_size"]
        k = max(1, min(int(k), total_n))
        return all_idx[:k], None

    pool = min(config["al_candidate_pool"], total_n)
    candidate_indices = random.sample(range(total_n), pool) if pool < total_n else list(range(total_n))
    scores = compute_information_scores(ddpm, dataset, candidate_indices, config)

    k = max(1, min(int(config["al_subset_size"]), total_n))
    n_explore = int(config["al_explore_ratio"] * k)
    n_explore = max(0, min(n_explore, k))
    n_greedy = k - n_explore

    order = np.argsort(-scores)
    greedy_pick = [candidate_indices[i] for i in order[:min(n_greedy, len(order))]]
    remaining = [candidate_indices[i] for i in order[min(n_greedy, len(order)):]]

    if n_explore > 0:
        explore_pick = random.sample(remaining, min(n_explore, len(remaining))) if len(remaining) > 0 else []
    else:
        explore_pick = []

    selected = greedy_pick + explore_pick

    if len(selected) < k:
        rest = list(set(range(total_n)) - set(selected))
        if len(rest) > 0:
            selected += random.sample(rest, min(k - len(selected), len(rest)))

    random.shuffle(selected)
    al_log = {"candidate_indices": candidate_indices, "candidate_scores": scores}
    return selected[:k], al_log


def select_indices_blockwise(ddpm, dataset, epoch, config):
    """
    Block-wise active sampling with a fixed subset size.

    Behavior:
    - First block: random subset
    - Each new block afterward: active sampling selection
    - The selected subset is cached and reused within the block
    """
    total_n = len(dataset)
    block_epochs = int(config.get("al_block_epochs", 5))
    k = int(config.get("al_subset_size_fixed", 3000))
    k = max(1, min(k, total_n))

    if "al_cached_indices" not in config:
        config["al_cached_indices"] = None
        config["al_cached_block_id"] = -1
        config["al_cached_log"] = None

    block_id = epoch // block_epochs

    if (block_id == config["al_cached_block_id"]) and (config["al_cached_indices"] is not None):
        return config["al_cached_indices"], config["al_cached_log"], block_id, "cache"

    if block_id == 0 and config.get("al_first_block_random", True):
        all_idx = list(range(total_n))
        random.shuffle(all_idx)
        selected_indices = all_idx[:k]
        al_log = None
        mode = "random_first_block"
    else:
        cfg = {**config, "al_subset_size": k}
        selected_indices, al_log = select_active_indices(ddpm, dataset, epoch, cfg)
        mode = "active_sampling_refresh"

    config["al_cached_indices"] = selected_indices
    config["al_cached_block_id"] = block_id
    config["al_cached_log"] = al_log

    print(f"Block selection | block_id={block_id}, mode={mode}, subset_size={len(selected_indices)}")
    return selected_indices, al_log, block_id, mode


def save_al_log(al_log, save_dir, epoch):
    if al_log is None:
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "idx": al_log["candidate_indices"],
        "score": al_log["candidate_scores"],
    }).sort_values("score", ascending=False)

    df.to_csv(
        os.path.join(save_dir, f"al_scores_epoch_{epoch + 1:04d}.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 5) Training
# ============================================================
def train_ddpm_with_active_sampling(config):
    print("Starting DDPM training with Active Sampling...")

    dataset = XLSXImageDataset(
        root_dir=config["root_dir"],
        image_size=config["image_size"],
        deform_min=config["deform_min"],
        deform_max=config["deform_max"],
        polygon_points=config.get("polygon_points", None),
        sheet_name=config.get("sheet_name", "Sheet1"),
        skiprows=config.get("skiprows", 1),
        multiply_minus_one=config.get("multiply_minus_one", False),
    )

    net = UNet(in_channels=config["in_channels"], n_feat=config["n_feat"])
    ddpm = DDPM(net, betas=config["betas"], n_T=config["n_T"], device=config["device"]).to(config["device"])

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config["lrate"], weight_decay=1e-4)
    scaler = GradScaler("cuda")

    os.makedirs(config["save_dir"], exist_ok=True)
    al_log_dir = os.path.join(config["save_dir"], "active_sampling_logs")
    epoch_log_xlsx = os.path.join(config["save_dir"], "training_epoch_log.xlsx")
    epoch_rows = []

    for epoch in range(config["n_epoch"]):
        ddpm.train()
        epoch_start = time.perf_counter()

        al_scoring_time_sec = 0.0

        if config.get("enable_active_sampling", False):
            t_al0 = time.perf_counter()
            selected_indices, al_log, block_id, mode = select_indices_blockwise(ddpm, dataset, epoch, config)
            al_scoring_time_sec = time.perf_counter() - t_al0

            if al_log is not None:
                save_al_log(al_log, al_log_dir, epoch)

            train_ds = Subset(dataset, selected_indices)
            dataloader = DataLoader(
                train_ds,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config.get("num_workers", 8),
                pin_memory=True,
                drop_last=True,
                persistent_workers=(config.get("num_workers", 0) > 0),
            )
            subset_len = len(train_ds)

            print(
                f"Epoch {epoch + 1}: block_id={block_id}, mode={mode}, "
                f"subset={subset_len}/{len(dataset)}"
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config.get("num_workers", 8),
                pin_memory=True,
                drop_last=True,
                persistent_workers=(config.get("num_workers", 0) > 0),
            )
            subset_len = len(dataset)

        n_batches = len(dataloader)
        effective_train_samples = n_batches * config["batch_size"]

        print(
            f"Epoch {epoch + 1}: batches={n_batches}, batch_size={config['batch_size']}, "
            f"effective_train_samples={effective_train_samples}"
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config["n_epoch"] * len(dataloader)),
        )

        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['n_epoch']}")

        for x in pbar:
            x = x.to(config["device"], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                loss = ddpm(x)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), config.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{epoch_loss / (pbar.n + 1):.4f}"})

        epoch_time = time.perf_counter() - epoch_start
        mean_loss = epoch_loss / max(1, len(dataloader))
        lr = float(optimizer.param_groups[0]["lr"])

        epoch_rows.append({
            "epoch": epoch + 1,
            "mean_loss": float(mean_loss),
            "lr": lr,
            "epoch_time_sec": float(epoch_time),
            "al_scoring_time_sec": float(al_scoring_time_sec),
            "subset_len": int(subset_len),
            "dataset_len": int(len(dataset)),
            "effective_train_samples": int(effective_train_samples),
        })

        flush_epoch_log_to_xlsx(epoch_rows, epoch_log_xlsx, sheet_name="epoch_log")

        if config.get("save_model", True) and ((epoch + 1) % config.get("save_every_epochs", 5) == 0):
            ckpt_path = os.path.join(config["save_dir"], f"ddpm_epoch_{epoch + 1}.pth")
            torch.save(ddpm.state_dict(), ckpt_path)
            print(f"Saved: {ckpt_path}")

    if config.get("save_model", True):
        final_path = os.path.join(config["save_dir"], "ddpm_final.pth")
        torch.save(ddpm.state_dict(), final_path)
        print(f"Saved final model: {final_path}")

    return ddpm


# ============================================================
# 6) Main
# ============================================================
if __name__ == "__main__":
    print("Program start: DDPM training with Active Sampling")
    seed_everything(42)

    config = {
        # ---------------- data ----------------
        "root_dir": r"I:\GenerationSample\20250123-Sample_3-3",
        "save_dir": r"I:\Generation\2-Training\20260126-3-3",

        "image_size": (128, 128),
        "in_channels": 1,
        "sheet_name": "Sheet1",
        "skiprows": 1,

        # Set to True if the deformation values should be multiplied by -1
        "multiply_minus_one": False,

        # Manual global normalization range
        "deform_min": -0.7487,
        "deform_max": 6.4238,

        # Polygon mask points
        "polygon_points": [
            [-471.98878, 2661.68383],
            [-6.24321, 2878.80846],
            [8.65906, 2877.58484],
            [421.55126, 2678.54260],
            [421.55126, 2663.85916],
            [411.61641, 2650.26337],
            [377.04315, 2632.18099],
            [89.52871, 2584.73171],
            [-70.62101, 2583.64405],
            [-282.03453, 2594.92854],
            [-473.57836, 2654.88594],
        ],

        # ---------------- DDPM ----------------
        "n_T": 200,
        "betas": (1e-4, 0.05),
        "n_feat": 256,
        "device": "cuda:0",

        # ---------------- training ----------------
        "n_epoch": 30,
        "batch_size": 8,
        "lrate": 5e-4,
        "num_workers": 12,
        "grad_clip": 1.0,
        "save_model": True,
        "save_every_epochs": 5,

        # ---------------- Active Sampling ----------------
        "enable_active_sampling": True,

        "al_block_epochs": 5,
        "al_subset_size_fixed": 3000,
        "al_first_block_random": True,

        "al_warmup_epochs": 0,
        "al_candidate_pool": 5000,
        "al_score_batch_size": 8,
        "al_score_num_workers": 0,
        "al_score_n_repeat": 2,
        "al_explore_ratio": 0.20,

        # ---------------- run mode ----------------
        "run_training": True,
        "load_ckpt_path": r"I:\Generation\2-Training\20260126-3-3\ddpm_final.pth",
    }

    net = UNet(in_channels=config["in_channels"], n_feat=config["n_feat"])
    ddpm = DDPM(
        net,
        betas=config["betas"],
        n_T=config["n_T"],
        device=config["device"],
    ).to(config["device"])

    if config.get("run_training", True):
        ddpm = train_ddpm_with_active_sampling(config)
    else:
        load_ckpt_path = config.get("load_ckpt_path", "")
        if (load_ckpt_path is None) or (len(str(load_ckpt_path)) == 0):
            raise RuntimeError(
                "run_training=False but load_ckpt_path is empty. "
                "Please set config['load_ckpt_path']."
            )

        if not os.path.exists(load_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_ckpt_path}")

        load_checkpoint(load_ckpt_path, ddpm, map_location=config["device"])

    print("All done.")