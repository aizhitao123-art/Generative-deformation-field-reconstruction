# Generative-deformation-field-reconstruction
DDPM-based deformation field reconstruction with Active Sampling training and point-guided conditional inference for sparse monitoring data.
This repository provides a Denoising Diffusion Probabilistic Model (DDPM) framework for deformation field reconstruction from sparse monitoring data. It includes:



\- \*\*Training\*\* with block-wise \*\*Active Sampling\*\*

\- \*\*Inference\*\* with \*\*point-guided conditional sampling\*\*

\- Support for \*\*XLSX-based spatial deformation data\*\*

\- Export of generated samples to \*\*scatter PNG\*\* and \*\*Excel files\*\*

\- Point-wise error evaluation at monitoring locations



The code is designed for deformation field generation and reconstruction tasks where spatial data are represented by scattered `(X, Y, value)` samples and converted into raster images for diffusion modeling.



---



\## Files



This repository currently contains two main scripts:



\- `train\_ddpm\_active\_sampling.py`  

&nbsp; DDPM training script with Active Sampling



\- `infer\_ddpm\_points\_guided.py`  

&nbsp; DDPM inference script with point-guided conditional sampling



You may rename the files as needed, but the README assumes the names above.



---



\## Features



\### Training

\- XLSX-based dataset loading

\- Conversion from scattered points to raster images

\- Global normalization to `\[-1, 1]`

\- Time-conditioned U-Net backbone

\- Standard DDPM training

\- Block-wise Active Sampling for efficient subset selection

\- Checkpoint saving and epoch logging



\### Inference

\- Compatible with multiple checkpoint formats

\- Point-guided conditional sampling using observation-consistency gradients

\- Reconstruction under sparse monitoring constraints

\- Scatter cloud PNG export

\- Optional Excel export of generated deformation fields

\- Point-wise error statistics including:

&nbsp; - MAE

&nbsp; - RMSE

&nbsp; - Mean relative error



---



\## Data Format



\### Training data

The training script expects a directory containing multiple `.xlsx` files.



Each XLSX file should contain a sheet with at least three columns:



1\. `X` coordinate

2\. `Y` coordinate

3\. deformation value



Example:



| X | Y | Value |

|---|---|-------|

| ... | ... | ... |



These scattered samples are mapped onto a 2D raster grid.



\### Inference points

The inference script uses monitoring points in the following format:



```python

points = \[

&nbsp;   {"x": ..., "y": ..., "v": ...},

&nbsp;   {"x": ..., "y": ..., "v": ...},

]

