# -*- coding: utf-8 -*-
"""
DDPM Inference / Conditional Sampling (Point-Guided) + Scatter Cloud PNG

Compatible with both checkpoint formats:
1) torch.save(ddpm.state_dict(), "*.pth")
2) torch.save({"model_state": ddpm.state_dict(), ...}, "*.pth")

Features:
- Conditional sampling with observation-consistency gradient guidance
- Output scatter PNG figures
- Optional XLSX export
- Point-wise error evaluation table

Plot style:
- blue-white-red custom colormap
- fixed vmin / vmax
- equal axis scaling
- configurable DPI
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, Polygon


# ============================================================
# 0) Colormap
# ============================================================
custom_cmap = LinearSegmentedColormap.from_list(
    name="blue_white_red",
    colors=["#4C7CB8", "#FFFCBB", "#D73527"],
    N=256
)


# ============================================================
# 1) Random seed
# ============================================================
def seed_everything(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) Model definition
# ============================================================
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            out = (x + x2) if self.same_channels else (x1 + x2)
            return out / 1.414
        return self.conv2(self.conv1(x))


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.block1 = ResidualConvBlock(out_channels + skip_channels, out_channels)
        self.block2 = ResidualConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, input_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_feat: int = 256) -> None:
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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


def ddpm_schedules(beta1: float, beta2: float, t_steps: int):
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
    def __init__(self, nn_model: nn.Module, betas, n_T: int, device: str) -> None:
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.n_T = int(n_T)
        self.device = device

        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v)

    @torch.no_grad()
    def sample(self, n_sample: int, size, device: str) -> torch.Tensor:
        self.eval()
        x_i = torch.randn(n_sample, *size, device=device)
        for i in range(self.n_T, 0, -1):
            t_norm = torch.full((n_sample, 1), i / self.n_T, device=device)
            z = torch.randn_like(x_i) if i > 1 else 0
            eps = self.nn_model(x_i, t_norm)
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i

    def _x0_from_xt_eps(self, x_t: torch.Tensor, eps: torch.Tensor, i: int) -> torch.Tensor:
        return (x_t - self.sqrtmab[i] * eps) / (self.sqrtab[i] + 1e-12)

    def sample_with_points(
        self,
        n_sample: int,
        size,
        device: str,
        points,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        deform_min: float,
        deform_max: float,
        obs_weight: float = 1.0,
        obs_lr: float = 0.05,
        clamp_x0: bool = True,
    ) -> torch.Tensor:
        """
        Point-guided conditional sampling using observation-consistency guidance.

        Args:
            points: list of dictionaries, each formatted as
                {"x": ..., "y": ..., "v": ...}
                where x and y are real coordinates and v is the real deformation value.
            deform_min / deform_max:
                Must be the same normalization range used during training.
        """
        self.eval()
        x_i = torch.randn(n_sample, *size, device=device)

        h, w = size[-2], size[-1]

        pixel_indices = []
        target_values = []

        for p in points:
            r, c = coord_to_ij(p["x"], p["y"], x_min, x_max, y_min, y_max, h, w)
            pixel_indices.append((r, c))
            target_values.append(norm_real_to_m11(p["v"], deform_min, deform_max))

        y_targets = torch.tensor(target_values, device=device, dtype=torch.float32).view(1, -1)

        for i in range(self.n_T, 0, -1):
            t_norm = torch.full((n_sample, 1), i / self.n_T, device=device, dtype=torch.float32)

            x_req = x_i.detach().requires_grad_(True)
            eps_g = self.nn_model(x_req, t_norm)
            x0_g = self._x0_from_xt_eps(x_req, eps_g, i)

            if clamp_x0:
                x0_g = x0_g.clamp(-1, 1)

            pred_list = [x0_g[:, 0, r, c] for (r, c) in pixel_indices]
            y_pred = torch.stack(pred_list, dim=1)

            loss_obs = F.mse_loss(y_pred, y_targets.expand_as(y_pred))
            grad = torch.autograd.grad(loss_obs, x_req, retain_graph=False, create_graph=False)[0]

            x_i = (x_req - obs_lr * obs_weight * grad).detach()

            with torch.no_grad():
                eps2 = self.nn_model(x_i, t_norm)

            z = torch.randn_like(x_i) if i > 1 else 0
            x_i = self.oneover_sqrta[i] * (x_i - eps2 * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

        return x_i


# ============================================================
# 3) Checkpoint loading
# ============================================================
def _extract_state_dict_from_ckpt(ckpt_obj):
    """
    Extract a model state_dict from different checkpoint formats.

    Supported formats:
    A) Pure state_dict
    B) {"model_state": state_dict}
    C) {"state_dict": state_dict}
    D) {"model": state_dict}
    """
    if isinstance(ckpt_obj, dict):
        for key in ["model_state", "state_dict", "model"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key], ckpt_obj
        return ckpt_obj, ckpt_obj
    return ckpt_obj, None


def load_ddpm_from_ckpt(
    ckpt_path: str,
    device: str,
    in_channels: int = 1,
    n_feat: int = 256,
    n_T: int = 200,
    betas=(1e-4, 0.05),
    strict: bool = False,
):
    net = UNet(in_channels=in_channels, n_feat=n_feat)
    ddpm = DDPM(net, betas=betas, n_T=n_T, device=device).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state, meta = _extract_state_dict_from_ckpt(ckpt)

    missing, unexpected = ddpm.load_state_dict(state, strict=strict)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"strict={strict} | missing={len(missing)} | unexpected={len(unexpected)}")

    if len(missing) > 0:
        print("Missing keys (head):", missing[:10], "..." if len(missing) > 10 else "")
    if len(unexpected) > 0:
        print("Unexpected keys (head):", unexpected[:10], "..." if len(unexpected) > 10 else "")

    ddpm.eval()
    return ddpm, meta


# ============================================================
# 4) Normalization and coordinate helpers
# ============================================================
def denorm_m11_to_real(arr_m11: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map values from [-1, 1] to [vmin, vmax]."""
    return (arr_m11 + 1.0) * 0.5 * (vmax - vmin) + vmin


def get_coords_grid(x_min: float, x_max: float, y_min: float, y_max: float, h: int, w: int):
    x_coords = np.linspace(x_min, x_max, w)
    y_coords = np.linspace(y_min, y_max, h)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    return x_grid, y_grid


def coord_to_ij(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    h: int,
    w: int,
):
    col = (x - x_min) / (x_max - x_min + 1e-12) * (w - 1)
    row = (y - y_min) / (y_max - y_min + 1e-12) * (h - 1)

    col = int(round(float(col)))
    row = int(round(float(row)))

    col = max(0, min(w - 1, col))
    row = max(0, min(h - 1, row))
    return row, col


def norm_real_to_m11(v_real: float, vmin: float, vmax: float) -> float:
    v01 = (v_real - vmin) / (vmax - vmin + 1e-12)
    v01 = float(np.clip(v01, 0.0, 1.0))
    return v01 * 2.0 - 1.0


# ============================================================
# 5) Saving outputs
# ============================================================
def save_samples_scatter_png_and_excel(
    samp_tensor: torch.Tensor,
    out_dir: str,
    base_name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    deform_min: float,
    deform_max: float,
    polygon_points=None,
    s: int = 5,
    dpi: int = 300,
    save_xlsx: bool = True,
    save_cloud: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "cloud_png")
    if save_cloud:
        os.makedirs(png_dir, exist_ok=True)

    arr = samp_tensor.detach().cpu().numpy()[:, 0, :, :]
    n_samples, h, w = arr.shape

    arr = np.clip(arr, -1.0, 1.0)
    arr_phys = denorm_m11_to_real(arr, deform_min, deform_max)

    x_grid, y_grid = get_coords_grid(x_min, x_max, y_min, y_max, h, w)
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)

    polygon = Polygon(polygon_points) if (polygon_points is not None and len(polygon_points) >= 3) else None
    vmin, vmax = float(deform_min), float(deform_max)

    xlsx_path = os.path.join(out_dir, f"{base_name}.xlsx")
    writer = pd.ExcelWriter(xlsx_path, engine="openpyxl") if save_xlsx else None

    for i in range(n_samples):
        deform_flat = arr_phys[i].reshape(-1)
        df = pd.DataFrame({"X": x_flat, "Y": y_flat, "Deformation": deform_flat})

        if polygon is not None:
            inside_mask = np.array([polygon.contains(Point(float(x), float(y))) for x, y in zip(df["X"], df["Y"])])
            df = df[inside_mask]

        if save_xlsx:
            df.to_excel(writer, sheet_name=f"sample_{i + 1:03d}", index=False)

        if save_cloud:
            png_path = os.path.join(png_dir, f"{base_name}_sample_{i + 1:03d}.png")
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(
                df["X"],
                df["Y"],
                c=df["Deformation"],
                cmap=custom_cmap,
                s=s,
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(sc, label="Value")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"{base_name} | sample {i + 1}/{n_samples}")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(png_path, dpi=dpi)
            plt.close()

    if save_xlsx:
        writer.close()
        print("Saved XLSX:", xlsx_path)
    if save_cloud:
        print("Saved cloud PNGs:", png_dir)


def evaluate_samples_at_points(
    samp_tensor: torch.Tensor,
    points,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    deform_min: float,
    deform_max: float,
    out_dir: str,
    base_name: str = "ddpm_infer",
    eps: float = 1e-12,
    print_table: bool = True,
    save_excel: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    arr = samp_tensor.detach().cpu().numpy()[:, 0, :, :]
    n_samples, h, w = arr.shape

    arr = np.clip(arr, -1.0, 1.0)
    arr_phys = denorm_m11_to_real(arr, deform_min, deform_max)

    ij = [coord_to_ij(p["x"], p["y"], x_min, x_max, y_min, y_max, h, w) for p in points]

    rows = []
    summary_rows = []

    for si in range(n_samples):
        abs_err_list = []
        rel_err_list = []

        for pi, p in enumerate(points):
            r, c = ij[pi]
            pred = float(arr_phys[si, r, c])
            true = float(p["v"])
            abs_err = pred - true
            rel_err = abs(abs_err) / (abs(true) + eps)

            abs_err_list.append(abs(abs_err))
            rel_err_list.append(rel_err)

            rows.append({
                "sample": si + 1,
                "point_id": pi + 1,
                "x": float(p["x"]),
                "y": float(p["y"]),
                "true_v": true,
                "pred_v": pred,
                "abs_err": abs(abs_err),
                "rel_err": rel_err,
                "pixel_row": int(r),
                "pixel_col": int(c),
            })

        abs_err_arr = np.array(abs_err_list, dtype=np.float64)
        rel_err_arr = np.array(rel_err_list, dtype=np.float64)

        mae = float(abs_err_arr.mean())
        rmse = float(np.sqrt((abs_err_arr ** 2).mean()))
        mape = float(rel_err_arr.mean())

        summary_rows.append({
            "sample": si + 1,
            "MAE(points)": mae,
            "RMSE(points)": rmse,
            "MeanRelErr(points)": mape,
        })

    df_points = pd.DataFrame(rows)
    df_summary = pd.DataFrame(summary_rows)

    avg_row = {
        "sample": "AVG",
        "MAE(points)": float(df_summary["MAE(points)"].mean()),
        "RMSE(points)": float(df_summary["RMSE(points)"].mean()),
        "MeanRelErr(points)": float(df_summary["MeanRelErr(points)"].mean()),
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)

    if print_table:
        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 200)
        print("\n========== Point-wise results (first 50 rows) ==========")
        print(df_points.head(50).to_string(index=False))
        print("\n========== Summary (per sample + AVG) ==========")
        print(df_summary.to_string(index=False))

    if save_excel:
        out_xlsx = os.path.join(out_dir, f"{base_name}_point_errors.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df_points.to_excel(writer, sheet_name="point_wise", index=False)
            df_summary.to_excel(writer, sheet_name="summary", index=False)
        print("Saved point error XLSX:", out_xlsx)

    return df_points, df_summary


def try_load_global_stats_json(stats_json_path: str):
    if (stats_json_path is None) or (not os.path.exists(stats_json_path)):
        return None
    with open(stats_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 6) Main
# ============================================================
if __name__ == "__main__":
    seed_everything(1)

    # --------------------------------------------------------
    # (A) Paths
    # --------------------------------------------------------
    ckpt_path = r"E:\Generation\2-Training\20260204-3-3\ddpm_final_3-3.pth"
    out_dir = r"E:\Generation\3-Sampling\ExpeI-Comparison-WithAS\AS\inference_samples_3-3_SEED1"

    # Optional JSON file containing coordinate bounds
    stats_json = r"E:\Generation\2-Training\20251226-3-3\global_stats.json"

    # --------------------------------------------------------
    # (B) Model settings
    # --------------------------------------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_size = (128, 128)
    in_channels = 1
    n_feat = 256
    n_T = 200
    betas = (1e-4, 0.05)

    # --------------------------------------------------------
    # (C) Physical normalization range
    # Must match training
    # --------------------------------------------------------
    deform_min = -0.7487
    deform_max = 6.4238

    # --------------------------------------------------------
    # (D) Coordinate range
    # --------------------------------------------------------
    stats = try_load_global_stats_json(stats_json)
    if stats is not None:
        x_min, x_max = float(stats["x_min"]), float(stats["x_max"])
        y_min, y_max = float(stats["y_min"]), float(stats["y_max"])
        print("Loaded coordinate bounds from stats JSON:", x_min, x_max, y_min, y_max)
    else:
        x_min, x_max = -471.98878, 421.55126
        y_min, y_max = 2583.64405, 2878.80846
        print("Using manual coordinate bounds:", x_min, x_max, y_min, y_max)

    # --------------------------------------------------------
    # (E) Polygon mask
    # --------------------------------------------------------
    polygon_points = [
        [-471.98878, 2661.68383],
        [-6.24321, 2878.80846],
        [8.65906, 2877.58484],
        [421.55126, 2678.5426],
        [421.55126, 2663.85916],
        [411.61641, 2650.26337],
        [377.04315, 2632.18099],
        [89.52871, 2584.73171],
        [-70.62101, 2583.64405],
        [-282.03453, 2594.92854],
        [-473.57836, 2654.88594]
    ]

    # --------------------------------------------------------
    # (F) Sampling configuration
    # --------------------------------------------------------
    n_sample = 8
    dpi = 300
    point_size = 5
    save_xlsx = True
    save_cloud = True

    print("Device:", device)
    print("Loading checkpoint:", ckpt_path)

    ddpm, meta = load_ddpm_from_ckpt(
        ckpt_path=ckpt_path,
        device=device,
        in_channels=in_channels,
        n_feat=n_feat,
        n_T=n_T,
        betas=betas,
        strict=False,
    )

    # --------------------------------------------------------
    # Monitoring points in real coordinates and real values
    # --------------------------------------------------------
    points = [
        {"x": -138.600006, "y": 2725, "v": 2.34},       # HL9
        {"x": -178.600006, "y": 2725, "v": 2.64},       # HL10
        {"x": 140.599884, "y": 2709, "v": 2.50},        # CH57
        {"x": 100.599884, "y": 2775, "v": 1.88},        # CH47
        {"x": 187.599884, "y": 2654.048828, "v": 1.47}, # DC5-5
        {"x": -178.016006, "y": 2646, "v": 1.85},       # DC3-6
    ]

    # --------------------------------------------------------
    # Conditional sampling
    # --------------------------------------------------------
    x_gen = ddpm.sample_with_points(
        n_sample=n_sample,
        size=(in_channels, *image_size),
        device=device,
        points=points,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        deform_min=deform_min,
        deform_max=deform_max,
        obs_weight=50.0,
        obs_lr=0.2,
        clamp_x0=True,
    )

    print(
        "Generated samples:",
        tuple(x_gen.shape),
        "range≈",
        float(x_gen.min().item()),
        float(x_gen.max().item())
    )

    base_name = "ddpm_points_guided"

    save_samples_scatter_png_and_excel(
        samp_tensor=x_gen,
        out_dir=out_dir,
        base_name=base_name,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        deform_min=deform_min,
        deform_max=deform_max,
        polygon_points=polygon_points,
        s=point_size,
        dpi=dpi,
        save_xlsx=save_xlsx,
        save_cloud=save_cloud,
    )

    evaluate_samples_at_points(
        samp_tensor=x_gen,
        points=points,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        deform_min=deform_min,
        deform_max=deform_max,
        out_dir=out_dir,
        base_name=base_name,
        print_table=True,
        save_excel=True,
    )

    print("Done. Output directory:", out_dir)