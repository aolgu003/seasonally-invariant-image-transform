# CLAUDE.md

## Project Overview

This repository contains the official training code for the **Science Robotics** paper: *"A seasonally invariant deep transform for visual terrain-relative navigation"* (Fragoso et al., 2021). It trains deep neural networks (U-Net) to produce image transformations that are invariant to seasonal appearance changes (e.g., summer vs. winter), enabling robust visual terrain-relative navigation for robots.

The project is being extended to support multi-modal image matching across:
- **Aerial thermal to satellite** matching
- **Aerial grayscale to satellite** matching
- **Frame-to-frame thermal** matching (temporal)
- **Frame-to-frame visible** matching (temporal)

Two training approaches are implemented:
- **NCC-based** (`siamese-ncc.py`): Optimizes for normalized cross-correlation registration
- **SIFT-based** (`siamese-sift.py`): Optimizes for SIFT feature matching (detector + descriptor losses)

## Repository Structure

```
├── siamese-ncc.py              # NCC-optimized training entry point
├── siamese-sift.py             # SIFT-optimized training entry point
├── siamese-inference.py        # Inference entry point
├── createTiledDataset.py       # Preprocessing: crops large images into tiles
├── requirements.txt            # Python dependencies (pip)
├── model/
│   ├── unet.py                 # U-Net architecture (1-ch in, 1-ch out, sigmoid output)
│   ├── unet_parts.py           # U-Net building blocks (DoubleConv, Down, Up, OutConv)
│   ├── correlator.py           # Normalized cross-correlation layer
│   ├── kornia_dog.py           # Difference of Gaussians multi-scale pyramid
│   └── kornia_sift.py          # SIFT descriptor extraction via Kornia
├── dataset/
│   ├── neg_dataset.py          # Siamese dataset for NCC training (with negative sampling)
│   ├── neg_sift_dataset.py     # Siamese dataset for SIFT training (with augmentation)
│   └── inference_dataset.py    # Dataset for inference
├── utils/
│   └── helper.py               # Normer, tensorboard logging, pyramid loss, inference helpers
├── data/
│   ├── coregistered_images/    # Raw paired images (on/ and off/ season subdirs)
│   └── samples/                # Sample images for inference demos
├── weights2try/                # Pre-trained model weights
│   ├── ctCo300dx1-weights/     # best_test_weights.pt, best_train_weights.pt
│   └── mtCo150ax1-weights/     # best_test_weights.pt, best_train_weights.pt
├── experiments/                # Output directory for experiment results and weights
└── runs/                       # TensorBoard log directory (gitignored)
```

## Tech Stack

- **Language**: Python 3
- **Deep Learning**: PyTorch 1.11.0, PyTorch Lightning 1.7.7
- **Computer Vision**: Kornia 0.6.6, OpenCV 4.5.5.64
- **Data Augmentation**: Albumentations 1.1.0
- **Image I/O**: Pillow, imageio, rasterio
- **Visualization**: TensorBoard, matplotlib
- **Scientific Computing**: NumPy, SciPy, scikit-image

## Environment Setup

```bash
# Create a Python 3 environment (conda recommended)
pip3 install -r requirements.txt
```

The `requirements.txt` contains pinned versions for all 207+ dependencies. Some packages are installed from local conda builds and git repos (e.g., detectron2, multipoint).

## Common Commands

### Data Preprocessing

Crop large orthorectified images into training tiles:

```bash
python createTiledDataset.py \
  --raw_data_dir=data/coregistered_images/off \
  --save_data_dir=data/training_pairs/off \
  --overlap_ratio=0 \
  --crop_width=600 \
  --crop_height=600

python createTiledDataset.py \
  --raw_data_dir=data/coregistered_images/on \
  --save_data_dir=data/training_pairs/on \
  --overlap_ratio=0 \
  --crop_width=600 \
  --crop_height=600
```

### Training (NCC)

```bash
python siamese-ncc.py \
  --exp_name=correlation-toy-example \
  --training_data_dir=data/training_pairs/ \
  --validation_data_dir=data/training_pairs/ \
  --batch-size=4 \
  --epochs=100 \
  --device=0 \
  --num_workers=4
```

### Training (SIFT)

```bash
python siamese-sift.py \
  --exp_name=sift-toy-example \
  --training_data_dir=data/training_pairs/ \
  --validation_data_dir=data/training_pairs/ \
  --subsamples=100 \
  --crop_width=64 \
  --batch-size=2 \
  --zeta=10 \
  --gamma=1 \
  --epochs=100
```

Key SIFT-specific args:
- `--zeta`: Descriptor loss weight (default: 10)
- `--gamma`: Detector loss weight (default: 1)
- `--subsamples`: Number of random crop subsamples per batch (default: 100)
- `--crop_width`: Crop size for pyramid loss patches (default: 64)

### Inference

```bash
python siamese-inference.py \
  --data_dir=data/samples/fakeplaceid_fakequad_000015_on.png \
  --output_dir=correlation-toy-example/sample-outputs \
  --weights_path=correlation-toy-example/weights/best_test_weights.pt \
  --mean=0.4 \
  --std=0.12
```

### TensorBoard

```bash
tensorboard --logdir=runs/
```

## Architecture Details

### Model

- **U-Net** with 1 grayscale input channel and 1 output channel
- Encoder: 5 levels (64 -> 128 -> 256 -> 512 -> 512 channels)
- Decoder: 4 upsampling levels with skip connections
- Bilinear upsampling (default), transpose convolution available
- Sigmoid activation on output (produces [0, 1] range images)
- Based on [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

### Training Pipeline

1. **Data**: Paired coregistered "on-season" and "off-season" images, tiled into patches
2. **Dataset**: Siamese pairs with 50% negative (mismatched) samples
3. **Loss (NCC)**: MSE between predicted NCC score and label (1=match, 0=mismatch)
4. **Loss (SIFT)**: Weighted combination of DoG pyramid correlation loss and SIFT descriptor distance loss: `gamma * sift_loss + zeta * pyramid_loss`
5. **Optimizer**: Adam (lr=1e-5) with ExponentialLR scheduler (gamma=0.995 for NCC, 0.99 for SIFT)
6. **Outputs**: Best test/train weights saved to `experiments/{exp_name}/weights/`

### Data Format

- **Input**: PNG images (grayscale or RGB converted to grayscale)
- **Directory structure**: Paired images in `on/` and `off/` subdirectories with matching filenames
- **Normalization stats**:
  - NCC dataset: on-season mean=0.49, std=0.12; off-season mean=0.44, std=0.10
  - SIFT dataset: on-season mean=0.49, std=0.135; off-season mean=0.44, std=0.12
  - Inference defaults: mean=0.4, std=0.12

## Key Conventions

- Random seeds are fixed for reproducibility (torch.manual_seed(0), random.seed(0), np.random.seed(0))
- `torch.autograd.set_detect_anomaly(True)` is enabled in training scripts for debugging
- `cudnn.deterministic = True` and `cudnn.benchmark = False` for reproducibility
- Weights are saved as PyTorch state dicts (`.pt` files)
- Experiment outputs go to `experiments/{exp_name}/` and TensorBoard logs to `runs/{exp_name}/`
- Mixed precision training via NVIDIA Apex is supported but optional (auto-detected)

## Testing

There is no formal test suite. Validation is performed during training via the validation dataset split. Verify model quality by:

1. Monitoring TensorBoard loss curves during training
2. Running inference and visually inspecting transformed images
3. Comparing NCC scores or SIFT feature matches between transformed on/off season pairs

## Code Style

- No linter or formatter configuration files are present (black is installed but unconfigured)
- No pre-commit hooks
- No CI/CD pipelines
- Commit messages use imperative mood, lowercase, brief descriptions

## Notes for AI Assistants

- This is research code accompanying a published paper. The original seasonal matching functionality must remain working throughout all refactoring.
- The SIFT-based training is more sensitive to hyperparameters than NCC. The README warns about tuning `--zeta` and `--gamma`.
- The `data/training_pairs/` directory is gitignored; it must be generated from `data/coregistered_images/` via `createTiledDataset.py`.
- Pre-trained weights are available in `weights2try/` for quick inference experiments.
- GPU is assumed for training (`--device=0`); CPU fallback exists but will be slow.

## Known Issues

- **`siamese-sift.py` line 69**: `input1` and `input2` are both assigned from `data[0]`, so `data[1]` (the off-season image) is never used during SIFT training. This is likely a bug.
- **`createTiledDataset.py` line 17**: `h, w, c = np.asarray(img).shape` assumes 3D arrays and crashes on grayscale PNGs loaded as 2D.
- **`correlator.py` line 20**: Random noise (`torch.randn`) is added during normalization, which introduces non-determinism at inference time.
- **Hardcoded normalization stats differ** between NCC dataset (std=0.12/0.10) and SIFT dataset (std=0.135/0.12) with no documentation explaining why.

---

## Roadmap: Multi-Modal Support, Testing, and CI/CD

### Goal

Extend the codebase to support four new matching scenarios beyond seasonal on/off matching, while adding the testing infrastructure and CI/CD pipelines needed to verify existing and new functionality automatically.

**New matching scenarios:**
1. Aerial thermal image to satellite image
2. Aerial grayscale image to satellite image
3. Frame-to-frame matching for thermal video
4. Frame-to-frame matching for visible video

### Phase 1: Testing Infrastructure and Bug Fixes

Establish a test foundation before making any functional changes, so that refactoring can proceed with confidence.

**1.1 Add pytest and testing tooling**
- Add `pytest`, `pytest-cov`, `pytest-mock` to dependencies
- Create `tests/` directory structure mirroring `model/`, `dataset/`, `utils/`
- Configure `pyproject.toml` with pytest settings and coverage thresholds

**1.2 Write unit tests for existing code (current behavior)**
- `tests/model/test_unet.py` — forward pass shapes, sigmoid output range, channel config
- `tests/model/test_correlator.py` — NCC score range [-1, 1], matching vs non-matching pairs, batch handling
- `tests/model/test_kornia_dog.py` — pyramid output structure, scale count, shape consistency
- `tests/model/test_kornia_sift.py` — descriptor shape (N, 128), keypoint extraction, single-channel assertion
- `tests/dataset/test_neg_dataset.py` — pair loading, negative sampling ratio, normalization application
- `tests/dataset/test_neg_sift_dataset.py` — augmentation pipeline, pair integrity
- `tests/dataset/test_inference_dataset.py` — single-image loading, normalization
- `tests/utils/test_helper.py` — Normer output properties, pyramid_loss shape, write_tensorboard calls
- `tests/test_create_tiled_dataset.py` — tile dimensions, overlap math, edge handling

**1.3 Fix known bugs**
- Fix `siamese-sift.py` line 69: use `data[1]` for second input
- Fix `createTiledDataset.py` line 17: handle 2D grayscale arrays
- Fix `correlator.py` line 20: remove random noise addition (use `torch.clamp` on std instead)

**1.4 Add linting and formatting**
- Configure `black` (already installed) and `ruff` in `pyproject.toml`
- Add `pre-commit` config with black, ruff, and trailing whitespace hooks

### Phase 2: Configuration System

Replace hardcoded values with a configuration system so that different modality pairs can each specify their own parameters without code changes.

**2.1 Introduce YAML-based experiment configs**
- Create `configs/` directory with per-scenario config files:
  - `configs/seasonal_ncc.yaml` — current on/off season NCC defaults
  - `configs/seasonal_sift.yaml` — current on/off season SIFT defaults
  - `configs/thermal_satellite.yaml`
  - `configs/grayscale_satellite.yaml`
  - `configs/frame_to_frame_thermal.yaml`
  - `configs/frame_to_frame_visible.yaml`
- Each config specifies: modality names, channel counts, normalization stats, loss weights, learning rate, scheduler params, SIFT thresholds, scale pyramid params

**2.2 Externalize all hardcoded parameters**
- Normalization stats (mean/std per modality) — currently hardcoded in `neg_dataset.py:50-51`, `neg_sift_dataset.py:56-57`, `siamese-inference.py:57-58`
- Learning rate and scheduler gamma — currently hardcoded in `siamese-ncc.py:99-101`, `siamese-sift.py:274-276`
- SIFT thresholds (contrastThreshold, edgeThreshold, nOctaveLayers, numFeatures) — currently hardcoded in `siamese-sift.py:271`
- Matching thresholds (scale_dist < 2, center_dist <= 5) — currently hardcoded in `siamese-sift.py:184-185`
- Scale pyramid params (n_levels, init_sigma, min_size) — currently hardcoded in `siamese-sift.py:261-265`

**2.3 Add a stats computation utility**
- Script to compute per-channel mean/std from a dataset directory
- Cache results alongside the dataset or in the config file
- Eliminate the need to manually specify normalization stats

**2.4 Tests for the config system**
- Config loading/validation, missing field errors, default fallback behavior

### Phase 3: Generalize the Data Pipeline

Decouple the dataset classes from the "on/off season" directory convention so they can handle arbitrary modality pairs and temporal sequences.

**3.1 Replace directory-convention pairing with a manifest system**
- Support a JSON manifest that explicitly lists pairs with modality labels:
  ```json
  {
    "modality_a": {"name": "thermal", "channels": 1, "dir": "thermal/"},
    "modality_b": {"name": "satellite", "channels": 3, "dir": "satellite/"},
    "pairs": [
      {"a": "scene_001.tif", "b": "scene_001.tif"},
      ...
    ]
  }
  ```
- Fall back to current on/off directory scanning when no manifest is provided (backward compat)
- Replace `img_on_path.replace('/on/', '/off/')` (neg_dataset.py:27) with manifest-driven lookup

**3.2 Support multi-channel image loading**
- Replace `.convert('L')` (neg_dataset.py:46-47, inference_dataset.py:23) with channel-aware loading
- Replace `cv2.imread(path, 0)` (neg_sift_dataset.py:47-48) with configurable channel flag
- Support `.tif`/`.tiff` (thermal, satellite multispectral) and `.jp2` in addition to `.png`
- Handle 16-bit image data (common in thermal and satellite)

**3.3 Add temporal/frame-to-frame dataset mode**
- Dataset variant that pairs consecutive frames (frame_t, frame_t+1) from a sequence directory
- Support configurable temporal stride (pair every N frames)
- Sequence-aware splitting: ensure train/val splits don't leak across sequences

**3.4 Per-channel normalization**
- Move normalization stats into config (from hardcoded values)
- Support per-channel stats for multi-spectral inputs
- Separate normalization paths for each modality in a pair (thermal and satellite have very different distributions)

**3.5 Fix createTiledDataset.py for multi-modal use**
- Handle 2D (grayscale) and 3D (color/multispectral) arrays
- Support `.tif` input alongside `.png`
- Add paired tiling mode: tile both modalities in lockstep so tile indices correspond

**3.6 Tests for the data pipeline**
- Manifest loading and validation
- Multi-channel image loading (1-ch, 3-ch, 16-bit)
- Frame-to-frame pairing with stride
- Backward compatibility with on/off directory structure
- Tiling consistency across modality pairs

### Phase 4: Generalize the Model and Matching Layers

Make the model architecture and correlation layers work with variable channel counts and matching strategies.

**4.1 Parameterize UNet input/output channels**
- Already supports `n_channels` and `n_classes` arguments; the change is in the training scripts which hardcode `n_channels=1, n_classes=1`
- Read channel config from YAML and pass through
- Make sigmoid output activation optional (thermal may need linear output)

**4.2 Update Correlator for multi-channel and spatial matching**
- Make epsilon configurable (thermal needs larger epsilon due to noise)
- Remove `torch.randn` noise injection; use `torch.clamp(std, min=eps)` instead
- Add optional spatial correlation map output (for frame-to-frame dense matching)

**4.3 Update SIFT pipeline for multi-modal inputs**
- Remove `assert(PC == 1)` in `kornia_sift.py:53` — allow multi-channel or convert internally
- Make num_features, patch_size, contrastThreshold, edgeThreshold configurable from YAML
- Make `laf_from_opencv_kpts` mrSize configurable (currently hardcoded to 6.0 in `siamese-sift.py:28`)

**4.4 Update DoG pyramid for configurable scales**
- Expose `n_levels`, `init_sigma`, `min_size`, `double_image` as config parameters
- Different modalities need different scale space settings (thermal features are coarser)

**4.5 Tests for model components**
- UNet: forward pass with 1, 3, and 4 input channels; with/without sigmoid
- Correlator: deterministic output (no random noise), correct score range, multi-channel inputs
- SIFT: descriptor extraction with configurable params, multi-channel handling
- DoG: pyramid output shapes with different config parameters
- End-to-end: load pre-trained weights, run inference, verify output shape and value range

### Phase 5: Update Training Scripts

Refactor the training entry points to use the config system and support all modality combinations.

**5.1 Unify training entry points**
- Create a single `train.py` that accepts `--config` and dispatches to NCC or SIFT loss based on config
- Keep `siamese-ncc.py` and `siamese-sift.py` as thin wrappers for backward compatibility
- All hyperparameters come from config YAML, with CLI overrides

**5.2 Generalize the training loop**
- Remove "on"/"off" terminology from variable names — use "modality_a"/"modality_b" or "source"/"target"
- Support asymmetric transforms (different U-Net per modality, or shared encoder with modality-specific heads)
- Add checkpoint resume support (save/load optimizer state, epoch, best loss)

**5.3 Update inference script**
- Read model channel config from the weights file or a companion config
- Support multi-channel output saving
- Remove hardcoded 2000x2000 image size cap (make configurable)

**5.4 Integration tests for training**
- Smoke test: 2-epoch training run with tiny synthetic data for each config
- Verify weights are saved, loss decreases, TensorBoard logs are written
- Test checkpoint resume produces same results

### Phase 6: CI/CD Pipeline

Automate verification so that every change is tested before merge.

**6.1 GitHub Actions workflow: lint and unit tests**
- Trigger on push and PR to main
- Steps: install deps, run ruff/black check, run `pytest tests/ --cov` with coverage threshold
- Use a lightweight CPU-only PyTorch install for CI speed

**6.2 GitHub Actions workflow: integration tests**
- Trigger on PR to main
- Run training smoke tests with synthetic data (NCC + SIFT, 2 epochs each)
- Run inference with pre-trained weights and verify output
- Optional GPU runner for full training validation (can be manual trigger)

**6.3 GitHub Actions workflow: model regression**
- Trigger on PR to main (or manual)
- Run inference with pre-trained weights on sample images
- Compare output against stored reference outputs (pixel-level or NCC score threshold)
- Catches regressions where model architecture changes break pre-trained weight compatibility

**6.4 Pre-commit hooks**
- black formatting check
- ruff linting
- Trailing whitespace and file ending fixes
- Optional: pytest on changed files

**6.5 Dependency management**
- Replace the monolithic `requirements.txt` (207 packages, many from conda local builds) with:
  - `requirements.txt` — core runtime deps with version ranges
  - `requirements-dev.txt` — test/lint/CI deps
  - Consider `pyproject.toml` as single source of truth

### Phase 7: Multi-Modal Feature Work

With infrastructure in place, implement the actual new matching scenarios.

**7.1 Aerial thermal to satellite matching**
- Thermal images are single-channel, low texture, different dynamic range than visible satellite
- May require separate U-Net branches (thermal encoder + satellite encoder) or a channel-adapter layer before the shared U-Net
- Normalization: thermal has very different stats from visible satellite
- Loss tuning: NCC may work better than SIFT for thermal (fewer sharp features)
- Config: `configs/thermal_satellite.yaml`

**7.2 Aerial grayscale to satellite matching**
- Closer to the original seasonal use case (grayscale to grayscale)
- Main difference: perspective distortion between aerial and satellite views
- May need augmentation for viewpoint changes (affine/projective transforms)
- Config: `configs/grayscale_satellite.yaml`

**7.3 Frame-to-frame thermal matching**
- Temporal sequences, not pre-registered pairs
- Need temporal dataset loader (consecutive frames)
- Smaller appearance change between frames than between seasons
- May want optical-flow-style spatial correlation maps instead of scalar NCC
- Config: `configs/frame_to_frame_thermal.yaml`

**7.4 Frame-to-frame visible matching**
- Similar to thermal frame-to-frame but with RGB/grayscale visible data
- More texture available; SIFT-based approach likely more effective
- Config: `configs/frame_to_frame_visible.yaml`

**7.5 Evaluation metrics**
- Add quantitative evaluation beyond TensorBoard loss curves:
  - NCC score distribution on held-out pairs
  - SIFT match count and inlier ratio (RANSAC)
  - Registration accuracy (pixel error after alignment)
- Store metrics per experiment for cross-modality comparison

### Implementation Order

The recommended order minimizes risk by building infrastructure before features:

| Order | Phase | Rationale |
|-------|-------|-----------|
| 1 | Phase 1: Testing + bug fixes | Establish safety net before any changes |
| 2 | Phase 6.4-6.5: Pre-commit + deps | Low effort, immediate developer experience improvement |
| 3 | Phase 2: Config system | Foundation that all subsequent work depends on |
| 4 | Phase 3: Data pipeline | Unblocks all new modality work |
| 5 | Phase 4: Model generalization | Enables multi-channel and spatial matching |
| 6 | Phase 5: Training script unification | Clean entry point for all experiments |
| 7 | Phase 6.1-6.3: CI/CD pipelines | Automate everything built so far |
| 8 | Phase 7: Multi-modal features | New functionality on solid foundation |
