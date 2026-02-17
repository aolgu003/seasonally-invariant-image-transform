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

---

## BDD Behavior Catalog: Existing Features

This catalog documents every testable behavior in the current codebase using BDD-style Given/When/Then specifications. Each feature maps to a source file and the behaviors describe what the code actually does today, including edge cases and known bugs. Tests should be written against these behaviors first to lock in a safety net before any refactoring.

---

### Feature: UNet Image Transformation (`model/unet.py`, `model/unet_parts.py`)

The U-Net transforms an input image into a seasonally-invariant representation.

**Scenario: Forward pass preserves spatial dimensions**
- Given a UNet with `n_channels=1` and `n_classes=1`
- When a tensor of shape `(B, 1, H, W)` is passed through the model
- Then the output shape is `(B, 1, H, W)` (same as input)

**Scenario: Output is bounded by sigmoid activation**
- Given any input tensor
- When passed through the UNet
- Then all output values are in the range `[0, 1]`

**Scenario: Bilinear mode uses bicubic upsampling with reflect-padded convolutions**
- Given a UNet with `bilinear=True`
- Then the `Up` blocks use `nn.Upsample(scale_factor=2, mode='bicubic')` followed by a `Conv2d` with `padding_mode='reflect'`

**Scenario: Transpose convolution mode uses ConvTranspose2d**
- Given a UNet with `bilinear=False`
- Then the `Up` blocks use `nn.ConvTranspose2d` for upsampling

**Scenario: Skip connections handle size mismatches**
- Given encoder and decoder feature maps with mismatched spatial dimensions (odd-sized inputs)
- When the `Up` block concatenates them
- Then the decoder feature map is padded to match the encoder feature map before concatenation

**Scenario: Encoder produces expected channel progression**
- Given a UNet initialized with default settings
- Then the encoder produces features with channel counts `[64, 128, 256, 512, 512]` at each level

**Scenario: DoubleConv applies BatchNorm and ReLU**
- Given a `DoubleConv` block
- When a tensor passes through it
- Then it passes through `Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU`

**Scenario: Down block halves spatial dimensions**
- Given a `Down` block
- When a tensor of shape `(B, C, H, W)` passes through it
- Then the output has shape `(B, C_out, H/2, W/2)`

---

### Feature: Normalized Cross-Correlation (`model/correlator.py`)

The Correlator computes NCC scores between pairs of image embeddings.

**Scenario: Matching embeddings produce high correlation**
- Given two identical embedding tensors of shape `(B, C, H, W)`
- When the Correlator computes their correlation
- Then the output score is close to `1.0`

**Scenario: Uncorrelated embeddings produce low correlation**
- Given two random, uncorrelated embedding tensors
- When the Correlator computes their correlation
- Then the output score is close to `0.0`

**Scenario: Output is a scalar per batch element**
- Given two tensors of shape `(B, C, H, W)`
- When passed through the Correlator
- Then the output shape is `(B, 1, 1)`

**Scenario: Zero-mean normalization is applied before correlation**
- Given an embedding tensor
- When `normalize_batch_zero_mean` is called
- Then the output has approximately zero mean per channel

**Scenario: Random noise is injected during normalization (known issue)**
- Given an embedding tensor
- When `normalize_batch_zero_mean` is called multiple times
- Then results differ due to `torch.randn` noise addition on line 20

**Scenario: DSIFT normalization flattens across all dimensions**
- Given a tensor of shape `(B, H, W, F)` and `dsift=True`
- When `normalize_batch_zero_mean_dsift` is called
- Then normalization is computed across the flattened `H*W*F` dimension per batch element

**Scenario: Correlation is normalized by spatial-channel volume**
- Given two embeddings of shape `(B, C, H, W)`
- When `match_corr2` computes the correlation
- Then the raw dot product is divided by `H * W * C`

---

### Feature: Difference of Gaussians Pyramid (`model/kornia_dog.py`)

The DoG module builds a multi-scale pyramid and computes differences.

**Scenario: KorniaDoG produces multi-octave DoG outputs**
- Given an input tensor of shape `(B, 1, H, W)`
- When passed through `KorniaDoG`
- Then it returns a list of DoG tensors (one per octave), along with sigmas, distances, and pyramid tensors

**Scenario: DoG is computed as difference of adjacent pyramid levels**
- Given a scale pyramid with `n_levels` levels per octave
- When DoG is computed
- Then each octave's DoG has `n_levels - 1` layers (adjacent level differences)

**Scenario: KorniaDoGScalePyr uses different dimension ordering**
- Given the same input
- When passed through `KorniaDoGScalePyr` instead of `KorniaDoG`
- Then the DoG subtraction operates on `dim=1` (levels) instead of `dim=2`

**Scenario: Spatial dimensions decrease across octaves**
- Given a multi-octave pyramid
- Then each successive octave has approximately half the spatial dimensions of the previous

---

### Feature: SIFT Descriptor Extraction (`model/kornia_sift.py`)

The KorniaSift module detects keypoints and extracts 128-dimensional SIFT descriptors.

**Scenario: Descriptors have 128 dimensions**
- Given an input tensor and detected keypoints
- When SIFT descriptors are extracted
- Then each descriptor has exactly 128 elements: output shape `(B, N_keypoints, 128)`

**Scenario: Keypoints are detected when no LAF is provided**
- Given an input tensor and `laf=None`
- When `KorniaSift.forward` is called
- Then keypoints are detected using `ScaleSpaceDetector` and LAFs are returned

**Scenario: Pre-computed LAFs bypass detection**
- Given an input tensor and pre-computed `laf`
- When `KorniaSift.forward` is called
- Then detection is skipped and descriptors are extracted at the provided LAF locations

**Scenario: Single-channel input is required**
- Given an input with `PC != 1` in the extracted patches
- When descriptors are extracted
- Then an `AssertionError` is raised at line 53

**Scenario: Patch extraction uses fixed 32x32 patches**
- Given detected keypoints
- When patches are extracted from the pyramid
- Then each patch has size `32x32` (hardcoded `PS=32`)

---

### Feature: NCC Siamese Dataset (`dataset/neg_dataset.py`)

The dataset loads paired on/off season images with negative sampling.

**Scenario: Paired images are loaded from on/ and off/ directories**
- Given a `data_root` with `on/` and `off/` subdirectories containing matching PNG filenames
- When the dataset is initialized
- Then each `on/` image has a verified corresponding `off/` image

**Scenario: Initialization fails if pairs are incomplete**
- Given an `on/` image with no matching `off/` image
- When the dataset is initialized
- Then an `AssertionError` is raised (line 16)

**Scenario: Negative sampling produces mismatched pairs**
- Given `negative_weighting=0.5`
- When `__getitem__` is called
- Then approximately 50% of samples return mismatched pairs with `target=0`

**Scenario: Positive samples return matched pairs**
- Given a positive sample (no negative swap)
- When `__getitem__` is called
- Then it returns the matched on/off pair with `target=1`

**Scenario: Negative samples never return the same index**
- Given a negative sample is selected
- When a random replacement index is chosen
- Then it loops until `rand_index != index`

**Scenario: Images are converted to grayscale**
- Given any input image (RGB or grayscale)
- When loaded by the dataset
- Then it is converted to single-channel via `img.convert('L')`

**Scenario: Hardcoded normalization is applied**
- Given a loaded image pair
- When normalization is applied
- Then on-season: `(img - 0.49) / 0.12`, off-season: `(img - 0.44) / 0.10`

**Scenario: Dataset size is controlled by samples_to_use**
- Given `samples_to_use=0.5` and 100 total images
- When `__len__` is called
- Then it returns 50

**Scenario: samples_to_use greater than 1 is rejected**
- Given `samples_to_use=1.5`
- When the dataset is initialized
- Then an `AssertionError` is raised (line 21)

**Scenario: Output format is ((img_on, img_off), target)**
- When `__getitem__` is called
- Then it returns a tuple of `(img_on_tensor, img_off_tensor)` and an integer target

---

### Feature: SIFT Siamese Dataset (`dataset/neg_sift_dataset.py`)

The dataset loads paired images with augmentation for SIFT training.

**Scenario: Augmentation is applied consistently to both images**
- Given paired on/off images
- When augmentation is applied
- Then `RandomResizedCrop`, `HorizontalFlip`, `VerticalFlip`, and `RandomRotate90` are applied identically to both images via `additional_targets={'imageOff': 'image'}`

**Scenario: Crop dimensions match original image dimensions**
- Given images of size `(H, W)`
- When the augmentation transform is created
- Then `RandomResizedCrop` uses `height=H, width=W` from the first image in the dataset

**Scenario: Images are loaded as grayscale via OpenCV**
- Given an image path
- When loaded
- Then `cv2.imread(path, 0)` reads it as grayscale and divides by 255

**Scenario: Different normalization stats than NCC dataset**
- Given a loaded image pair
- When normalization is applied
- Then on-season: `(img - 0.49) / 0.135`, off-season: `(img - 0.44) / 0.12`

**Scenario: No negative sampling in dataset (done in training loop)**
- When `__getitem__` is called
- Then it always returns the matched pair `(img_on, img_off)` with no target label

**Scenario: Output tensors are float type**
- When `__getitem__` is called
- Then both `img_on` and `img_off` are explicitly cast to `.float()`

---

### Feature: Inference Dataset (`dataset/inference_dataset.py`)

The dataset loads single images for inference.

**Scenario: All PNGs in directory are loaded**
- Given a directory with PNG files
- When the dataset is initialized
- Then all `*.png` files are discovered via glob

**Scenario: Single file path is handled**
- Given a path to a single PNG file (not a directory)
- When the dataset is initialized
- Then `glob` returns just that one file

**Scenario: Images are normalized with configurable mean/std**
- Given `mean=0.4` and `std=0.12`
- When an image is loaded
- Then it is normalized as `(img - 0.4) / 0.12`

**Scenario: Image filename is returned alongside tensor**
- When `__getitem__` is called
- Then it returns `(tensor, basename)` where basename is the filename without directory

**Scenario: Default normalization uses different stats than training**
- Given no explicit mean/std
- Then defaults are `mean=0.5, std=0.1` (different from the training datasets' stats)

---

### Feature: Tile Creation (`createTiledDataset.py`)

The preprocessing script crops large images into training tiles.

**Scenario: Non-overlapping tiles cover the image**
- Given an image of size `(1200, 1800)` and `crop_width=600, crop_height=600, overlap_ratio=0`
- When tiles are created
- Then 6 tiles are produced (2 rows x 3 columns)

**Scenario: Overlapping tiles increase tile count**
- Given `overlap_ratio=0.5`
- When tiles are created
- Then tiles overlap by 50% and more tiles are produced per dimension

**Scenario: Edge pixels are discarded if not tile-aligned**
- Given an image of size `(700, 700)` and `crop_width=600, crop_height=600`
- When tiles are created
- Then only 1 tile is produced (the remaining 100px strip is discarded)

**Scenario: Tile filenames encode source image and index**
- Given a source image `foo.png`
- When tiles are created
- Then tiles are named `foo_000000.png`, `foo_000001.png`, etc.

**Scenario: Grayscale images crash (known bug)**
- Given a grayscale PNG loaded as a 2D array
- When `np.asarray(img).shape` is unpacked as `h, w, c`
- Then a `ValueError` is raised because 2D arrays don't have 3 dimensions

**Scenario: Output directory is created automatically**
- Given `save_data_dir` does not exist
- When the script runs
- Then the directory is created via `os.makedirs(exist_ok=True)`

---

### Feature: NCC Training Pipeline (`siamese-ncc.py`)

The NCC training script trains the U-Net for correlation-based matching.

**Scenario: Both images pass through the same UNet**
- Given a batch of paired images
- When the training step runs
- Then `output1 = model(input1)` and `output2 = model(input2)` use the same model (shared weights)

**Scenario: Loss is MSE between correlation score and label**
- Given UNet outputs and a label (0 or 1)
- When the loss is computed
- Then `loss = MSELoss(correlator(output1, output2), label)`

**Scenario: Best test weights are saved when test loss improves**
- Given a new test loss lower than the current best
- When the epoch completes
- Then `best_test_weights.pt` is saved to `experiments/{exp_name}/weights/`

**Scenario: Best train weights are saved when train loss improves**
- Given a new train loss lower than the current best
- When the epoch completes
- Then `best_train_weights.pt` is saved

**Scenario: Learning rate decays exponentially**
- Given the ExponentialLR scheduler with `gamma=0.995`
- When `scheduler.step()` is called each batch
- Then the learning rate is multiplied by 0.995

**Scenario: Metrics are logged to TensorBoard**
- Given a completed epoch
- When metrics are computed
- Then `train_loss` and `test_loss` scalars are written to `runs/{exp_name}/`

**Scenario: Empty dataloader returns zero metrics**
- Given a dataloader with 0 batches
- When `eval` runs
- Then it returns `np.zeros(1)`

**Scenario: Experiment output directory is created**
- Given an `exp_name`
- When training starts
- Then `experiments/{exp_name}/weights/` is created

---

### Feature: SIFT Training Pipeline (`siamese-sift.py`)

The SIFT training script trains the U-Net for feature-matching-based registration.

**Scenario: Both inputs use data[0] (known bug)**
- Given a batch from the SIFT dataset
- When `input1, input2 = data[0], data[0]` executes at line 69
- Then both inputs are the on-season image; the off-season image (`data[1]`) is never used

**Scenario: Loss combines pyramid correlation and SIFT descriptor distance**
- Given `gamma` and `zeta` weights
- When the loss is computed
- Then `loss = gamma * sift_loss + zeta * pyramid_loss`

**Scenario: Pyramid loss uses random subsampled crops**
- Given `subsamples=100` and `crop_width=64`
- When pyramid loss is computed
- Then 100 random crop pairs are sampled from random octaves and layers of the DoG pyramid

**Scenario: 50% of subsampled crops are negative (mismatched locations)**
- Given a subsample iteration
- When `random.random() < 0.5`
- Then the crop location for the second image is randomized and the target is set to 0

**Scenario: OpenCV SIFT detects keypoints on CPU**
- Given the UNet output scaled to `[0, 255]` byte format
- When `cv2_sift.detect` is called
- Then keypoints are detected on the CPU numpy array

**Scenario: Keypoints are padded to fixed count**
- Given fewer than `numFeatures=500` keypoints detected
- When `repeatListToLengthN` is called
- Then the keypoint list is repeated cyclically to reach exactly 500

**Scenario: Descriptor loss uses distance with margin**
- Given matched descriptors (positive) and unmatched descriptors (negative)
- When SIFT loss is computed
- Then `loss = positive_distance.mean() + ReLU(2 - negative_distance.mean())`

**Scenario: Keypoint matching uses scale and spatial thresholds**
- Given two sets of keypoints with LAFs
- When matching is determined
- Then pairs must satisfy `scale_dist < 2` AND `center_dist <= 5`

**Scenario: LAF conversion from OpenCV uses mrSize=6.0**
- Given OpenCV keypoints
- When `laf_from_opencv_kpts` converts them
- Then scales are computed as `mrSize * keypoint.size` with `mrSize=6.0`

**Scenario: Batches with no detected keypoints skip descriptor loss**
- Given no keypoints are detected in any batch element
- When the descriptor loss section runs
- Then `descriptor_positive`, `descriptor_negative`, and `descriptor_match_map` remain zero

**Scenario: NaN assertions guard against numerical instability**
- Given LAF scales, centers, distance matrices, and descriptors
- When computed
- Then assertions verify none contain NaN values (lines 172-203)

---

### Feature: Inference (`siamese-inference.py`)

The inference script applies a trained model to new images.

**Scenario: Model loads pre-trained weights**
- Given a weights file path
- When the model is initialized
- Then `model.load_state_dict(torch.load(weights_path))` loads the saved parameters

**Scenario: Output images are saved with original filenames**
- Given input image `foo.png`
- When inference completes
- Then the transformed image is saved as `{output_dir}/foo.png`

**Scenario: Model runs in eval mode with no gradients**
- When inference executes
- Then `model.eval()` is called inside `torch.no_grad()`

**Scenario: CPU fallback when no GPU available**
- Given no CUDA-capable GPU
- When the device is selected
- Then `torch.device("cpu")` is used

---

### Feature: Helper Utilities (`utils/helper.py`)

Utility functions for normalization, inference, and loss computation.

**Scenario: Normer produces zero-mean output**
- Given any input tensor
- When `Normer()` is called
- Then the output has approximately zero mean

**Scenario: Normer uses epsilon to prevent division by zero**
- Given a constant-value tensor (std=0)
- When `Normer()` is called
- Then it does not raise a division error (epsilon=1e-7 added inside std computation)

**Scenario: Normer epsilon is applied inside std (bug-like behavior)**
- Given a tensor `x`
- When `Normer()` is called
- Then it computes `std(x + 1e-7)` rather than `std(x) + 1e-7`, which shifts values before computing std

**Scenario: inference_img caps image size at 2000x2000**
- Given an image larger than 2000 pixels in either dimension
- When `inference_img` is called
- Then the image is cropped to at most 2000x2000 from the top-left corner

**Scenario: inference_img normalizes with explicit mean/std**
- Given `mean=0.4` and `std=0.12`
- When the image is preprocessed
- Then it is normalized as `(tensor - 0.4) / 0.12`

**Scenario: make_sure_path_exists creates nested directories**
- Given a path `a/b/c` that does not exist
- When `make_sure_path_exists` is called
- Then all directories are created

**Scenario: make_sure_path_exists is idempotent**
- Given a path that already exists
- When `make_sure_path_exists` is called
- Then no error is raised (catches `EEXIST`)

**Scenario: pyramid_loss accumulates across pyramid levels**
- Given two pyramids `p1` and `p2` with multiple levels
- When `pyramid_loss` is called
- Then the correlation loss is summed across all pyramid levels

**Scenario: pyramid_loss treats each channel as a separate image**
- Given pyramid tensors of shape `(B, C, H, W)`
- When `pyramid_loss` reshapes them
- Then they become `(B*C, 1, H, W)` and labels are repeated `C` times

**Scenario: pyramid_loss_mse computes MSE across pyramid levels**
- Given two pyramids
- When `pyramid_loss_mse` is called
- Then it sums `MSELoss(l1, l2)` for each corresponding level pair

**Scenario: normalize_batch uses ImageNet stats**
- Given a batch tensor
- When `normalize_batch` is called
- Then it normalizes using `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`

**Scenario: rgb2gray_batch averages channels**
- Given a batch of shape `(B, 3, H, W)`
- When `rgb2gray_batch` is called
- Then it returns `(B, 1, H, W)` by averaging across channels

**Scenario: write_tensorboard logs all label-metric pairs**
- Given labels `['train_loss', 'train_acc']` and metrics `[0.5, 0.9]`
- When `write_tensorboard` is called
- Then `writer.add_scalar` is called once per label-metric pair

---

### Feature: Reproducibility (`siamese-ncc.py`, `siamese-sift.py`)

Both training scripts fix random seeds for deterministic behavior.

**Scenario: Random seeds are fixed**
- When the training script is imported/run
- Then `torch.manual_seed(0)`, `random.seed(0)`, and `np.random.seed(0)` are set

**Scenario: CuDNN is configured for determinism**
- When the training script runs
- Then `cudnn.deterministic = True` and `cudnn.benchmark = False`

**Scenario: Anomaly detection is enabled**
- When the training script runs
- Then `torch.autograd.set_detect_anomaly(True)` is active

---

### Feature: Pre-trained Weight Compatibility (`weights2try/`)

Pre-trained weights must remain loadable after any model changes.

**Scenario: ctCo300dx1 weights load into default UNet**
- Given `weights2try/ctCo300dx1-weights/best_test_weights.pt`
- When `UNet(n_channels=1, n_classes=1, bilinear=True).load_state_dict(...)` is called
- Then all keys match and the model loads without error

**Scenario: mtCo150ax1 weights load into default UNet**
- Given `weights2try/mtCo150ax1-weights/best_test_weights.pt`
- When loaded into the default UNet
- Then all keys match and the model loads without error

**Scenario: Inference with pre-trained weights produces valid output**
- Given a loaded model with pre-trained weights and a sample image
- When inference runs
- Then the output is a valid image with values in `[0, 1]`
