# OSI SAF ERAS ML — ERA5 Wind Bias Correction with Machine Learning

[![OSI SAF](https://img.shields.io/badge/OSI%20SAF-EUMETSAT-blue)](https://osi-saf.eumetsat.int/)
[![License](https://img.shields.io/badge/license-see%20repo-lightgrey)]()

This repository provides the inference code and pre-trained model weights for correcting persistent stress-equivalent wind biases in ERA5 (ECMWF Reanalysis v5) sea surface wind forecasts using Fully-connected Feed-forward Neural Networks (FCNNs). It accompanies the technical report:

> **Makarova, E., Portabella, M., and Stoffelen, A.** (2026). *On the Use of Machine Learning to Correct NWP Model Sea Surface Wind Forecasts with Scatterometer (ERASTAR AI2)*. OSI SAF Visiting Scientist Activity Technical Report OSI_VSA24_01. EUMETSAT.

Related peer-reviewed publication:

> **Makarova, E., Portabella, M., and Stoffelen, A.** (2025). *Reduction of persistent stress-equivalent wind biases with machine learning and scatterometer data*. IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–11. doi: [10.1109/TGRS.2025.3586375](https://doi.org/10.1109/TGRS.2025.3586375)

---

## Overview

ERA5 surface winds exhibit geographically structured and persistent biases compared to scatterometer observations, linked to unresolved air–sea interactions, SST gradient effects, and ocean current influences. This project provides an ML-based approach to correct these biases — specifically for the stress-equivalent 10-meter wind (U10S) — **without** requiring scatterometer observations at inference time.

The FCNNs were trained on five years (2019–2022 and 2024) of ERA5 atmospheric/oceanic variables and CMEMS GLOBCURRENT surface current data, using Advanced Scatterometer (ASCAT) observations as the training target. Two model flavours are provided:

- **Monthly models**: one model per calendar month, trained on the target month ±15 days of data. These generally outperform the all-year model, especially in the tropics.
- **All-year model**: a single model trained on the full five-year dataset, with a sine/cosine encoding of the day-of-year as an additional input.

For detailed validation results, see the [technical report (OSI_VSA24_01)](https://github.com/quaxoc/osisaf_erasml).

---

## Repository Structure

```
osisaf_erasml/
├── collocations/       # Tools/scripts for collocating ERA5 and GLOBCURRENT inputs
├── data/               # Configuration or ancillary data files
├── inference/          # Main inference scripts, including generate_outputs_fnn.py
├── metadata/           # Normalization statistics and input variable metadata
├── model/              # Neural network architecture definition (FCNN)
├── weights/            # Pre-trained model weights (.pt files)
│   ├── monthly/        # 12 monthly models (each trained on month ±15 days)
│   └── allyear/        # All-year model with sincos day-of-year encoding
└── conda.yml           # Conda environment specification
```

---

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/quaxoc/osisaf_erasml.git
cd osisaf_erasml
conda env create -f conda.yml
conda activate osisaf_erasml   # adjust name if different in conda.yml
```

---

## Input Data Requirements

Before running inference, you need to prepare and collocate the following input fields on a **0.125° regular grid**:

| Variable | Source | Notes |
|---|---|---|
| Eastward / Northward U10S components | ERA5 | Stress-equivalent winds at 10 m |
| U10S speed and direction | Derived from ERA5 | |
| U10S curl and divergence | Derived from ERA5 |  |
| Mean sea level pressure | ERA5 | |
| Surface air temperature | ERA5 | |
| Specific humidity | ERA5 | |
| Sea surface temperature (SST) | ERA5 |  |
| Eastward / Northward SST gradients | Derived from ERA5 SST | |
| Eastward / Northward sea water velocity | CMEMS GLOBCURRENT | Daily mean |

ERA5 data are sourced from twice-daily analyses (06:00 and 18:00 UTC) with hourly forecast steps from +3 to +18 hours. GLOBCURRENT daily averages centered at 12:00 UTC are used.

See `collocations/` for the collocation tools and `metadata/` for normalization parameters.

---

## Running Inference

The script can be run from any directory:

```bash
python inference/generate_outputs_fnn.py 20230101 20231231
```

Before running, open `inference/generate_outputs_fnn.py` and review the configuration block at the top of the file. The key parameters to set are:

- `nwp_dir` / `currents_dir`: paths to your ERA5 and GLOBCURRENT input data
- `nwp_an`: analysis times to process (currently only 6 and 18 UTC are supported)
- `nwp_fc`: forecast steps to process in hours — defaults to FC+3 only; change to `np.arange(3, 19)` for the full range used in training
- `model_path`: path to the weights to load (see [Pre-trained Weights](#pre-trained-weights) below)
- `currents_prefix`: set to `"dataset-uv-rep-daily_"` for the reprocessed CMEMS product or `"dataset-uv-nrt-daily_"` for the near-real-time product

The script will:
1. Load and collocate ERA5 and GLOBCURRENT input fields onto the 0.125° grid.
2. Normalize inputs using the statistics in `metadata/`.
3. Run inference on the FCNN to predict bias corrections for both zonal (u) and meridional (v) wind components.
4. Apply the corrections to the ERA5 U10S forecasts.
5. Write output to NetCDF. GRIB output is also supported but requires the conversion block in the script to be uncommented (`cdo` is included in `conda.yml`).

> **All-year model**: when using the all-year weights, uncomment `'date'` in the `input_var_names` list in the configuration block. This enables the sine/cosine day-of-year encoding that the all-year model requires. The monthly models do not use this input.

---

## Model Architecture

Both model families use the same FCNN architecture, selected via hyperparameter search:

- **5 hidden layers**: 1024 → 512 → 256 → 128 → 64 neurons
- **Dropout** between hidden layers (rate: 0.15)
- **Weight decay**: 5×10⁻⁵
- **Activation**: ReLU (implicit in FCNN design)
- **Loss function**: RMSE
- **Optimizer**: Adam (learning rate: 10⁻⁴)
- **Input**: 17 features (see table above) + sincos(day-of-year) for the all-year model
- **Output**: 2 values — bias corrections for u and v wind components

Directional inputs (longitude, U10S direction, day-of-year) use sine/cosine encoding. Other variables are normalized by mean and standard deviation (see `metadata/` for values).

> **Note on spatial variance**: The point-by-point FCNN approach preserves the spatial variance of the corrected wind fields. CNN architectures (e.g., U-Net) were evaluated but discarded because RMSE/MAE loss functions caused excessive spatial smoothing.

---

## Pre-trained Weights

The `weights/` directory contains:

- **12 monthly model weight files** — one per calendar month, each trained on data from the target month ±15 days across 5 years (2019–2022, 2024). Recommended for most applications, especially in tropical regions.
- **1 all-year model weight file** — trained on the full 5-year dataset with sincos day-of-year encoding. Useful when a single unified model is preferred.

Training data: 15% subsample of 5 years of ASCAT-B/C collocations.
Validation data: 5% subsample of 2018 data.
Test period: full year 2023 (not used in training or validation).

---

## Citation

If you use this code or the pre-trained weights, please cite:

```bibtex
@techreport{makarova2026erastar,
  title     = {On the Use of Machine Learning to Correct {NWP} Model Sea Surface Wind Forecasts
               with Scatterometer ({ERASTAR\_AI2})},
  author    = {Makarova, Evgeniia and Portabella, Marcos and Stoffelen, Ad},
  year      = {2026},
  month     = {February},
  institution = {EUMETSAT Ocean and Sea Ice SAF},
  number    = {OSI\_VSA24\_01},
  type      = {Visiting Scientist Activity Technical Report}
}

@article{makarova2025reduction,
  title   = {Reduction of Persistent Stress-Equivalent Wind Biases with Machine Learning
             and Scatterometer Data},
  author  = {Makarova, Evgeniia and Portabella, Marcos and Stoffelen, Ad},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {63},
  pages   = {1--11},
  year    = {2025},
  doi     = {10.1109/TGRS.2025.3586375}
}
```

---

## Acknowledgements

This work was carried out within the EUMETSAT **Ocean and Sea Ice Satellite Application Facility (OSI SAF)** Visiting Scientist Activity programme. The authors are affiliated with the Barcelona Expert Center (BEC ICM-CSIC) and the Koninklijk Nederlands Meteorologisch Instituut (KNMI). ERA5 data were obtained from ECMWF. Surface current data were obtained from the Copernicus Marine Environment Monitoring Service (CMEMS).

---

## Contact

For questions about the code or methodology, please open a GitHub Issue or contact the authors via their institutional addresses at BEC ICM-CSIC.
