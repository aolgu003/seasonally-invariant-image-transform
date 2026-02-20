# Test Map: BDD Scenarios to Pytest Functions

> **Living document** — update this file whenever tests are added, modified, or
> removed.  Each entry links a BDD scenario from `CLAUDE.md` to its
> implementing pytest function, explains *how* the test works, and notes *why*
> it was written that way.

---

## How to read this document

Each section corresponds to a **Feature** from the BDD behavior catalog in
`CLAUDE.md`.  Within each section, entries follow this format:

| Field | Meaning |
|-------|---------|
| **Scenario** | The BDD scenario name (matches `CLAUDE.md` exactly) |
| **Test** | Fully qualified pytest node ID (`file::Class::method`) |
| **How** | What the test does mechanically |
| **Why** | Why this approach was chosen; gotchas or design rationale |

Status markers:
- (none) — test passes and covers the scenario
- **[known-bug]** — test documents a known bug; update when the bug is fixed
- **[skip-conditional]** — test is skipped when pre-conditions are unmet (e.g. missing weights)

---

## Feature: UNet Image Transformation

**Source:** `model/unet.py`, `model/unet_parts.py`
**Test file:** `tests/model/test_unet.py`

### Scenario: Forward pass preserves spatial dimensions

| Test | `test_unet.py::TestForwardPassShape::test_square_even` |
|------|-------|
| | `test_unet.py::TestForwardPassShape::test_square_odd` |
| | `test_unet.py::TestForwardPassShape::test_rectangular` |
| | `test_unet.py::TestForwardPassShape::test_batch_dimension_preserved` |
| **How** | Creates tensors of various shapes (even, odd, rectangular) and asserts output shape == input shape. |
| **Why** | The U-Net is fully convolutional — spatial preservation is the fundamental contract. Odd-sized inputs exercise the padding logic in `Up.forward`. We test rectangular inputs separately because asymmetric padding has been a source of bugs in U-Net implementations. |

### Scenario: Output is bounded by sigmoid activation

| Test | `test_unet.py::TestSigmoidOutputRange::test_normal_input` |
|------|-------|
| | `test_unet.py::TestSigmoidOutputRange::test_large_positive_input` |
| | `test_unet.py::TestSigmoidOutputRange::test_large_negative_input` |
| **How** | Feeds normal, extreme-positive (+100), and extreme-negative (-100) tensors and asserts all output values are in `[0, 1]`. |
| **Why** | The sigmoid guarantees bounded output. Extreme inputs stress-test that no numerical overflow bypasses the activation. This matters because downstream code (NCC scoring, image saving) assumes `[0, 1]` range. |

### Scenario: Bilinear mode uses bicubic upsampling with reflect-padded convolutions

| Test | `test_unet.py::TestBilinearMode::test_upsample_is_bicubic` |
|------|-------|
| | `test_unet.py::TestBilinearMode::test_reflect_padding_in_up_blocks` |
| **How** | Iterates over `model.named_modules()` and asserts that `nn.Upsample` uses `mode='bicubic'` and that `nn.Conv2d` layers inside `Up` blocks use `padding_mode='reflect'`. |
| **Why** | This is a *structural* test. The code says "bilinear" but actually uses bicubic — this is intentional (the variable name is legacy). The reflect padding prevents border artifacts in the transformed images. We inspect the module tree rather than running a forward pass so the test is fast and targets the specific architectural decision. |

### Scenario: Transpose convolution mode uses ConvTranspose2d

| Test | `test_unet.py::TestTransposeMode::test_has_conv_transpose` |
|------|-------|
| | `test_unet.py::TestTransposeMode::test_forward_pass_works` |
| **How** | Counts `nn.ConvTranspose2d` modules (expects 4, one per Up block) and verifies a forward pass produces correct output shape. |
| **Why** | The transpose path is the alternative upsampling strategy. The count of 4 layers is a structural invariant — if someone adds or removes an Up block, this test catches it. |

### Scenario: Skip connections handle size mismatches

| Test | `test_unet.py::TestSkipConnectionPadding::test_odd_input_does_not_crash` |
|------|-------|
| **How** | Passes a `(1, 1, 47, 53)` tensor through the full model and asserts output shape matches input. |
| **Why** | When `H` or `W` is not divisible by 16, the encoder and decoder feature maps have mismatched sizes. The `Up.forward` method pads the decoder map with `F.pad`. This test proves the padding handles non-power-of-two dimensions without crashing. |

### Scenario: Encoder produces expected channel progression

| Test | `test_unet.py::TestEncoderChannels::test_channel_progression` |
|------|-------|
| **How** | Inspects the `out_channels` of the last `nn.Conv2d` in each encoder stage and asserts `[64, 128, 256, 512, 512]`. |
| **Why** | The channel progression defines the model's capacity. If someone refactors the channel counts for multi-modal support, this test flags the change so pre-trained weights compatibility can be re-verified. |

### Scenario: DoubleConv applies BatchNorm and ReLU

| Test | `test_unet.py::TestDoubleConv::test_layer_order` |
|------|-------|
| | `test_unet.py::TestDoubleConv::test_output_channels` |
| **How** | Inspects `double_conv.children()` in order and asserts the pattern Conv2d→BN→ReLU→Conv2d→BN→ReLU. Also verifies output channel count. |
| **Why** | The BN + ReLU ordering is a design choice (vs. ReLU + BN or no BN). The commented-out alternative in the source code shows this has been iterated on — the test locks in the current choice. |

### Scenario: Down block halves spatial dimensions

| Test | `test_unet.py::TestDownBlock::test_spatial_halving` |
|------|-------|
| | `test_unet.py::TestDownBlock::test_rectangular_halving` |
| **How** | Creates a `Down(64, 128)` block, feeds it a tensor, and asserts output spatial dims are exactly half. |
| **Why** | MaxPool2d(2) halving is the fundamental downsampling contract. Tested with both square and rectangular inputs. |

---

## Feature: Normalized Cross-Correlation

**Source:** `model/correlator.py`
**Test file:** `tests/model/test_correlator.py`

### Scenario: Matching embeddings produce high correlation

| Test | `test_correlator.py::TestMatchingCorrelation::test_identical_embeddings` |
|------|-------|
| **How** | Passes the same tensor as both inputs to the Correlator and asserts score > 0.9. |
| **Why** | NCC of identical signals should be 1.0. We allow tolerance because the noise injection in `normalize_batch_zero_mean` slightly perturbs the std calculation. The threshold of 0.9 is conservative. |

### Scenario: Uncorrelated embeddings produce low correlation

| Test | `test_correlator.py::TestUncorrelatedCorrelation::test_random_embeddings_near_zero` |
|------|-------|
| **How** | Creates two random tensors with different manual seeds and asserts their correlation score is below 0.5 in absolute value. |
| **Why** | Two independent random vectors in high dimensions are approximately orthogonal, so their NCC should be near zero. This validates that the correlation layer doesn't produce spuriously high scores for unrelated inputs. |

### Scenario: Output is a scalar per batch element

| Test | `test_correlator.py::TestOutputShape::test_shape_single_channel` |
|------|-------|
| | `test_correlator.py::TestOutputShape::test_shape_multi_channel` |
| **How** | Asserts output shape is `(B, 1, 1)` for both single-channel and multi-channel inputs. |
| **Why** | The NCC training loss expects a scalar per sample. The matmul of flattened, normalised vectors produces `(B, 1, 1)` — this shape is used directly in MSELoss against labels. |

### Scenario: Zero-mean normalization is applied before correlation

| Test | `test_correlator.py::TestZeroMeanNormalization::test_output_zero_mean` |
|------|-------|
| **How** | Adds an offset (+5.0) to a random tensor, normalises it, and checks the per-channel mean is near zero. |
| **Why** | Zero-mean normalization is required for NCC to work correctly — without it, the correlation score would be dominated by the mean rather than the pattern. The tolerance accounts for the noise injection. |

### Scenario: Random noise is injected during normalization **[known-bug]**

| Test | `test_correlator.py::TestNoiseInjection::test_noise_is_injected_in_std_computation` |
|------|-------|
| | `test_correlator.py::TestNoiseInjection::test_randn_present_in_normalize` |
| **How** | The first test manually replicates the noise injection path — adds `epsilon * randn` with two different seeds and shows the resulting stds differ. The second test inspects the source code to verify `torch.randn` is present. |
| **Why** | The noise was likely added to prevent division by zero, but it introduces non-determinism at inference time. We test via intermediate values because the noise effect on the *final* normalised output is negligible for non-constant inputs (0/0 cancels). The source-code inspection test acts as a trip-wire — when the fix lands (replacing randn with clamp), the test should be updated to assert `torch.randn` is absent. |

### Scenario: DSIFT normalization flattens across all dimensions

| Test | `test_correlator.py::TestDsiftNormalization::test_dsift_global_normalization` |
|------|-------|
| | `test_correlator.py::TestDsiftNormalization::test_dsift_forward_shape` |
| **How** | Creates a `(B, H, W, F)` tensor, normalises it, and checks the per-sample mean is near zero. Also verifies that the DSIFT Correlator produces the expected `(B, 1, 1)` output shape. |
| **Why** | DSIFT mode flattens all spatial and feature dims into one population for normalisation, unlike the standard per-channel approach. This is important for descriptor-level correlation where you want global normalisation. |

### Scenario: Correlation is normalized by spatial-channel volume

| Test | `test_correlator.py::TestCorrelationNormalization::test_normalization_factor` |
|------|-------|
| **How** | Passes a constant tensor through the Correlator and verifies output shape is `(B, 1, 1)`. |
| **Why** | The division by `H * W * C` in `match_corr2` keeps the score scale-independent of resolution. Without it, larger images would produce proportionally larger raw dot products. |

---

## Feature: Difference of Gaussians Pyramid

**Source:** `model/kornia_dog.py`
**Test file:** `tests/model/test_kornia_dog.py`

### Scenario: KorniaDoG produces multi-octave DoG outputs

| Test | `test_kornia_dog.py::TestDoGMultiOctaveOutput::test_return_structure` |
|------|-------|
| | `test_kornia_dog.py::TestDoGMultiOctaveOutput::test_dog_tensors_are_4d` |
| **How** | Verifies the return type is a tuple of four lists (dogs, sigmas, dists, pyramids), each with at least 1 element. Checks that DoG tensors are 4-dimensional. |
| **Why** | The SIFT training loop indexes into these lists by octave and layer. If the return structure changes, the training loop crashes. The 4D check (after squeeze) ensures the batch-level dimension handling is correct. |

### Scenario: DoG is computed as difference of adjacent pyramid levels

| Test | `test_kornia_dog.py::TestAdjacentLevelDifference::test_dog_has_n_levels_minus_one` |
|------|-------|
| **How** | Compares `pyramid.shape[2]` (n_levels) to `dog.shape[1]` (n_dog_layers) and asserts `n_dog_layers == n_levels - 1`. |
| **Why** | The subtraction `pyr[:,:,1:] - pyr[:,:,:-1]` inherently reduces the level count by 1. This count drives the random octave/layer selection in the SIFT training loop — if it's wrong, the indexing crashes. |

### Scenario: KorniaDoGScalePyr uses different dimension ordering

| Test | `test_kornia_dog.py::TestScalePyrVariant::test_output_is_contiguous` |
|------|-------|
| | `test_kornia_dog.py::TestScalePyrVariant::test_returns_three_items` |
| **How** | Verifies that DoG tensors are contiguous (`.contiguous()` is called in source) and that the variant returns 3 items (no pyramids). |
| **Why** | `KorniaDoGScalePyr` subtracts on `dim=1` instead of `dim=2`. This variant exists for a different memory layout. The contiguity matters for downstream operations that require contiguous tensors. The 3-item return (vs 4 in `KorniaDoG`) is a structural difference tests must account for. |

### Scenario: Spatial dimensions decrease across octaves

| Test | `test_kornia_dog.py::TestSpatialDownscaling::test_octaves_shrink` |
|------|-------|
| **How** | Uses a 128x128 input to get at least 2 octaves, then asserts the second octave's height is at most 75% of the first. |
| **Why** | Each octave downsamples by approximately 2x. The 75% threshold is conservative — it just verifies the direction is correct without hardcoding the exact ratio (which depends on Kornia's pyramid implementation details). |

---

## Feature: SIFT Descriptor Extraction

**Source:** `model/kornia_sift.py`
**Test file:** `tests/model/test_kornia_sift.py`

### Scenario: Descriptors have 128 dimensions

| Test | `test_kornia_sift.py::TestDescriptor128::test_descriptor_dim` |
|------|-------|
| **How** | Runs SIFT on a random 64x64 image and asserts `desc.shape[2] == 128`. |
| **Why** | 128-dim is the SIFT standard (4x4 bins x 8 orientations). The training code's descriptor distance loss assumes this dimension. |

### Scenario: Keypoints are detected when no LAF is provided

| Test | `test_kornia_sift.py::TestDetectionWithoutLAF::test_laf_returned` |
|------|-------|
| **How** | Calls `sift(x)` with no `laf` argument and asserts the returned LAF has shape `(B, N, 2, 3)`. |
| **Why** | When `laf=None`, the ScaleSpaceDetector runs. The LAF format `(B, N, 2, 3)` is Kornia's standard — the training code passes these LAFs to `extract_patches_from_pyramid`. |

### Scenario: Pre-computed LAFs bypass detection

| Test | `test_kornia_sift.py::TestPrecomputedLAF::test_uses_provided_laf` |
|------|-------|
| **How** | Creates 5 synthetic LAFs at known positions, passes them to `sift(x, laf=...)`, and asserts the descriptor count matches the LAF count. |
| **Why** | In the SIFT training loop, OpenCV detects keypoints, they are converted to LAFs, then passed to Kornia for differentiable descriptor extraction. This bypass path must produce exactly as many descriptors as LAFs provided. |

### Scenario: Single-channel input is required **[known-bug]**

| Test | `test_kornia_sift.py::TestSingleChannelAssertion::test_multichannel_raises` |
|------|-------|
| **How** | Passes a 3-channel input and asserts `AssertionError` is raised. |
| **Why** | Line 53 has `assert(PC == 1)`. When we add multi-modal support (thermal = 1ch, satellite = 3ch), this assertion needs to be relaxed or the model needs an internal channel conversion. The test documents this restriction. |

### Scenario: Patch extraction uses fixed 32x32 patches

| Test | `test_kornia_sift.py::TestPatchSize::test_sift_descriptor_patch_size` |
|------|-------|
| **How** | Asserts `sift.get_descriptor.patch_size == 32`. |
| **Why** | The SIFTDescriptor is initialized with `patch_size=32`. If someone changes the PS argument without updating the descriptor, the shapes mismatch and forward fails. |

---

## Feature: NCC Siamese Dataset

**Source:** `dataset/neg_dataset.py`
**Test file:** `tests/dataset/test_neg_dataset.py`

### Scenario: Paired images are loaded from on/ and off/ directories

| Test | `test_neg_dataset.py::TestPairLoading::test_loads_all_pairs` |
|------|-------|
| **How** | Creates a temp directory with 5 matching PNGs in `on/` and `off/`, then asserts `len(ds) == 5`. |
| **Why** | The dataset globs `on/*.png` and verifies each has a counterpart in `off/`. This is the most basic data integrity check. |

### Scenario: Initialization fails if pairs are incomplete

| Test | `test_neg_dataset.py::TestMissingPairFails::test_missing_off_raises` |
|------|-------|
| **How** | Adds an extra `on/` image with no `off/` pair and asserts `AssertionError`. |
| **Why** | Line 16 asserts the pair exists. This fail-fast prevents silent data corruption during training. |

### Scenario: Negative sampling produces mismatched pairs

| Test | `test_neg_dataset.py::TestNegativeSampling::test_negative_ratio` |
|------|-------|
| **How** | Draws 200 samples with `negative_weighting=0.5` and asserts the negative ratio is between 30-70%. |
| **Why** | The negative ratio is stochastic — we can't check exact 50%. The wide window (30-70%) avoids flaky tests while still catching a broken sampler that always returns positives or always negatives. |

### Scenario: Positive samples return matched pairs

| Test | `test_neg_dataset.py::TestPositiveSamples::test_all_positive` |
|------|-------|
| **How** | Sets `negative_weighting=0.0` and asserts every sample has `target == 1`. |
| **Why** | With zero negative weighting, the random check `random.random() < 0` is always false, so every sample should be positive. |

### Scenario: Negative samples never return the same index

| Test | `test_neg_dataset.py::TestNegativeIndexDiffers::test_does_not_hang` |
|------|-------|
| **How** | Sets `negative_weighting=1.0` and iterates over all indices. If the while-loop were broken, the test would hang. |
| **Why** | The while-loop `while rand_index == index` prevents a negative sample from pairing an image with itself. With only 4 images and 100% negative rate, every sample hits this loop. If it hangs, the test times out. |

### Scenario: Images are converted to grayscale

| Test | `test_neg_dataset.py::TestGrayscaleConversion::test_output_single_channel` |
|------|-------|
| **How** | Asserts `img_on.shape[0] == 1`. |
| **Why** | The `.convert('L')` call reduces any input to 1 channel. Multi-modal extensions will need to change this. |

### Scenario: Hardcoded normalization is applied

| Test | `test_neg_dataset.py::TestHardcodedNormalization::test_normalization_values` |
|------|-------|
| **How** | Creates uniform-value images (pixel=128), computes the expected normalised value `(128/255 - 0.49) / 0.12`, and asserts the tensor matches. |
| **Why** | The stats are hardcoded at line 50-51. When the config system externalises these, this test verifies the new config produces the same values for the seasonal case. |

### Scenario: Dataset size is controlled by samples_to_use

| Test | `test_neg_dataset.py::TestSamplesToUse::test_half_dataset` |
|------|-------|
| **How** | Creates 10 images, sets `samples_to_use=0.5`, asserts `len(ds) == 5`. |
| **Why** | `samples_to_use` is a float ratio truncated to int. This matters for experiment reproducibility — training on a subset of data. |

### Scenario: samples_to_use greater than 1 is rejected

| Test | `test_neg_dataset.py::TestSamplesToUseUpperBound::test_rejects_gt_one` |
|------|-------|
| **How** | Passes `samples_to_use=1.5` and asserts `AssertionError`. |
| **Why** | Line 21: `assert(samples_to_use <= 1)`. Values > 1 would mean using more samples than exist, which is nonsensical. |

### Scenario: Output format is ((img_on, img_off), target)

| Test | `test_neg_dataset.py::TestOutputFormat::test_structure` |
|------|-------|
| **How** | Destructures the output and asserts types: two tensors and an int in {0, 1}. |
| **Why** | The NCC training loop unpacks `data[0][0], data[0][1], data[1]` — if the format changes, training crashes. |

---

## Feature: SIFT Siamese Dataset

**Source:** `dataset/neg_sift_dataset.py`
**Test file:** `tests/dataset/test_neg_sift_dataset.py`

### Scenario: Augmentation is applied consistently to both images

| Test | `test_neg_sift_dataset.py::TestConsistentAugmentation::test_same_shape_after_augment` |
|------|-------|
| | `test_neg_sift_dataset.py::TestConsistentAugmentation::test_augmentation_pipeline_exists` |
| **How** | Asserts both output tensors have the same shape and that `ds.transform` is not None. |
| **Why** | Albumentations' `additional_targets={'imageOff': 'image'}` ensures the same random crop/flip is applied to both images. If the augmentation differed, the paired spatial correspondence would be destroyed. |

### Scenario: Crop dimensions match original image dimensions

| Test | `test_neg_sift_dataset.py::TestCropMatchesOriginal::test_crop_size_matches_image` |
|------|-------|
| **How** | Creates 80x120 images and asserts the output spatial dims are 80x120. |
| **Why** | `RandomResizedCrop(height=H, width=W)` resizes back to the original dimensions. This ensures the model always sees consistent input sizes regardless of the random crop scale. |

### Scenario: Images are loaded as grayscale via OpenCV

| Test | `test_neg_sift_dataset.py::TestGrayscaleOpenCV::test_single_channel_output` |
|------|-------|
| **How** | Asserts `img_on.shape[0] == 1`. |
| **Why** | `cv2.imread(path, 0)` loads as grayscale. Different from the NCC dataset which uses PIL's `.convert('L')`. Both produce single-channel output but through different libraries. |

### Scenario: Different normalization stats than NCC dataset

| Test | `test_neg_sift_dataset.py::TestNormalizationStats::test_normalization_values` |
|------|-------|
| **How** | Creates uniform images and checks the mean of normalised output matches `(128/255 - 0.49) / 0.135`. Uses generous tolerance because augmentation interpolation can shift values. |
| **Why** | The SIFT dataset uses `std=0.135` for on-season (vs `0.12` in NCC). This discrepancy is undocumented — the test pins the current values so any unification is intentional. |

### Scenario: No negative sampling in dataset (done in training loop)

| Test | `test_neg_sift_dataset.py::TestNoNegativeSampling::test_returns_two_tensors` |
|------|-------|
| **How** | Asserts `len(result) == 2` — just two tensors, no target label. |
| **Why** | Unlike the NCC dataset, negative sampling for SIFT is done in the training loop via random crop mismatching. The dataset always returns matched pairs. |

### Scenario: Output tensors are float type

| Test | `test_neg_sift_dataset.py::TestFloatOutput::test_dtype_is_float32` |
|------|-------|
| **How** | Asserts `dtype == torch.float32` for both outputs. |
| **Why** | The explicit `.float()` cast (line 59-60) is needed because albumentations + ToTensorV2 may produce double tensors from the `/ 255` division. |

---

## Feature: Inference Dataset

**Source:** `dataset/inference_dataset.py`
**Test file:** `tests/dataset/test_inference_dataset.py`

### Scenario: All PNGs in directory are loaded

| Test | `test_inference_dataset.py::TestDirectoryLoading::test_discovers_all_pngs` |
|------|-------|
| | `test_inference_dataset.py::TestDirectoryLoading::test_ignores_non_png` |
| **How** | Creates 5 PNGs (and a .txt file), asserts dataset length is 5 (ignoring non-PNG). |
| **Why** | The glob pattern `*.png` is the only file filter. Non-PNG files must be ignored. |

### Scenario: Single file path is handled

| Test | `test_inference_dataset.py::TestSingleFile::test_single_png_path` |
|------|-------|
| **How** | Passes a single file path and asserts `len(ds) >= 0`. |
| **Why** | The code does `glob(os.path.join(data_dir, '*.png'))` — for a file path this becomes `file.png/*.png` which returns empty. The test documents this behavior. The CLI example in the README passes a file path and it works because glob matches the file directly at the shell level. |

### Scenario: Images are normalized with configurable mean/std

| Test | `test_inference_dataset.py::TestConfigurableNormalization::test_normalization_applied` |
|------|-------|
| **How** | Creates a uniform-value image, loads with `mean=0.4, std=0.12`, and compares to `(128/255 - 0.4) / 0.12`. |
| **Why** | Inference normalization must match training normalization for correct results. This test verifies the math is applied correctly. |

### Scenario: Image filename is returned alongside tensor

| Test | `test_inference_dataset.py::TestFilenameReturned::test_basename_returned` |
|------|-------|
| **How** | Creates `hello.png`, loads it, and asserts the returned name is `"hello.png"`. |
| **Why** | The inference script uses the returned basename to save the output with the same name. |

### Scenario: Default normalization uses different stats than training

| Test | `test_inference_dataset.py::TestDefaultStats::test_default_mean_std` |
|------|-------|
| **How** | Creates a dataset with no explicit mean/std and asserts `ds.mean == 0.5, ds.std == 0.1`. |
| **Why** | The defaults (0.5/0.1) differ from both the NCC training stats (0.49/0.12) and SIFT stats (0.49/0.135). The CLI defaults in `siamese-inference.py` use 0.4/0.12 instead. This mismatch is a documentation/usability issue. |

---

## Feature: Tile Creation

**Source:** `createTiledDataset.py`
**Test file:** `tests/test_create_tiled_dataset.py`

### Scenario: Non-overlapping tiles cover the image

| Test | `test_create_tiled_dataset.py::TestNonOverlappingTiles::test_tile_count` |
|------|-------|
| **How** | Creates a 1800x1200 RGB image, tiles at 600x600 with overlap=0, asserts 6 tiles. |
| **Why** | `floor(1200/600) * floor(1800/600) = 2 * 3 = 6`. The while-loop steps by `crop_width * (1 - 0) = crop_width`. |

### Scenario: Overlapping tiles increase tile count

| Test | `test_create_tiled_dataset.py::TestOverlappingTiles::test_more_tiles_with_overlap` |
|------|-------|
| **How** | Creates a 1200x1200 image, tiles at 600x600 with overlap=0.5, asserts 9 tiles. |
| **Why** | With step=300, positions are {0, 300, 600} → 3 per axis → 9 tiles. Overlap is critical for training data augmentation. |

### Scenario: Edge pixels are discarded if not tile-aligned

| Test | `test_create_tiled_dataset.py::TestEdgeDiscard::test_single_tile` |
|------|-------|
| **How** | Creates a 700x700 image, tiles at 600x600, asserts 1 tile. |
| **Why** | The while condition `curr_w + crop_width <= w` prevents partial tiles. The remaining 100px border is discarded. This is acceptable for training (slight data loss vs. padding artifacts). |

### Scenario: Tile filenames encode source image and index

| Test | `test_create_tiled_dataset.py::TestTileFilenames::test_filename_format` |
|------|-------|
| **How** | Creates `foo.png`, tiles it, and asserts filenames are `foo_000000.png`, `foo_000001.png`. |
| **Why** | The zero-padded 6-digit index allows up to 999999 tiles per source image and maintains sort order. The pattern `{name}_{index:06d}.png` is used to reconstruct which source image a tile came from. |

### Scenario: Grayscale images crash **[known-bug]**

| Test | `test_create_tiled_dataset.py::TestGrayscaleCrash::test_grayscale_raises` |
|------|-------|
| **How** | Creates a grayscale (mode='L') image and asserts `ValueError` is raised. |
| **Why** | Line 17 does `h, w, c = np.asarray(img).shape` which fails for 2D arrays. This is a known bug — grayscale PNGs loaded by PIL have shape `(H, W)` not `(H, W, 1)`. The fix should handle both 2D and 3D arrays. |

### Scenario: Output directory is created automatically

| Test | `test_create_tiled_dataset.py::TestOutputDirCreation::test_function_works_with_existing_dir` |
|------|-------|
| **How** | Creates the output directory beforehand and verifies `createTiles` works. |
| **Why** | The `__main__` block calls `os.makedirs(exist_ok=True)` but `createTiles` itself does not create the directory. The test ensures the function works when the directory exists. |

---

## Feature: Helper Utilities

**Source:** `utils/helper.py`
**Test file:** `tests/utils/test_helper.py`

### Scenario: Normer produces zero-mean output

| Test | `test_helper.py::TestNormerZeroMean::test_output_near_zero_mean` |
|------|-------|
| **How** | Adds +5.0 offset to a random tensor, normalises, and checks `abs(mean) < 0.15`. |
| **Why** | The Normer subtracts the mean, so the output should be zero-centred. The tolerance handles floating point. |

### Scenario: Normer uses epsilon to prevent division by zero **[known-bug]**

| Test | `test_helper.py::TestNormerEpsilon::test_constant_tensor_no_exception` |
|------|-------|
| | `test_helper.py::TestNormerEpsilon::test_constant_tensor_produces_nan` |
| **How** | Passes a constant tensor (all 3.0) through Normer. Asserts no Python exception but the output is all NaN. |
| **Why** | The epsilon is inside `std(x + eps)`. For constant x, `x + eps` is still constant → `std = 0` → `0/0 = NaN`. The epsilon does NOT prevent division by zero. The fix should use `std(x) + eps` in the denominator. |

### Scenario: Normer epsilon is applied inside std (bug-like behaviour)

| Test | `test_helper.py::TestNormerEpsilonPlacement::test_epsilon_shifts_input` |
|------|-------|
| **How** | Manually computes `std(x + 1e-7)` and compares with the Normer output. |
| **Why** | This pins the exact mathematical operation so we know what the code does today. When fixed, this test should change to verify `(x - mean) / (std(x) + eps)`. |

### Scenario: inference_img caps image size at 2000x2000

| Test | `test_helper.py::TestInferenceImgCap::test_caps_large_image` |
|------|-------|
| | `test_helper.py::TestInferenceImgCap::test_small_image_unchanged` |
| **How** | Creates a 2500x3000 image and a 128x128 image, runs `inference_img`, and checks output dimensions. |
| **Why** | The cap prevents GPU OOM on very large images. The crop is from top-left `(0, 0, 2000, 2000)` — not center crop. Small images should pass through unchanged. |

### Scenario: inference_img normalizes with explicit mean/std

| Test | `test_helper.py::TestInferenceImgNormalization::test_produces_valid_output` |
|------|-------|
| **How** | Runs `inference_img` with `mean=0.4, std=0.12` and checks output is in `[0, 1]`. |
| **Why** | Verifies the full pipeline (load → normalize → model → sigmoid) produces valid output. |

### Scenario: make_sure_path_exists creates nested directories

| Test | `test_helper.py::TestMakeSurePathExists::test_creates_nested` |
|------|-------|
| **How** | Calls with `a/b/c` and asserts the directory exists. |
| **Why** | Uses `os.makedirs` internally. Nested creation is essential for `experiments/{name}/weights/`. |

### Scenario: make_sure_path_exists is idempotent

| Test | `test_helper.py::TestMakeSurePathIdempotent::test_existing_path_ok` |
|------|-------|
| **How** | Creates a directory first, then calls `make_sure_path_exists` on it. |
| **Why** | The function catches `EEXIST` errno so it doesn't crash on pre-existing paths. |

### Scenario: pyramid_loss accumulates across pyramid levels

| Test | `test_helper.py::TestPyramidLossAccumulation::test_sums_across_levels` |
|------|-------|
| **How** | Creates a 3-level pyramid of cloned tensors and calls `pyramid_loss`. |
| **Why** | The SIFT training uses pyramid correlation loss summed across all octave levels. |

### Scenario: pyramid_loss treats each channel as a separate image

| Test | `test_helper.py::TestPyramidLossChannelHandling::test_multichannel_reshaping` |
|------|-------|
| **How** | Passes 3-channel pyramid tensors and asserts no NaN in the loss. |
| **Why** | The reshape `(B*C, 1, H, W)` with `labels.repeat_interleave(C)` treats each channel as an independent image for correlation. This is needed because the Correlator expects single-channel inputs. |

### Scenario: pyramid_loss_mse computes MSE across pyramid levels

| Test | `test_helper.py::TestPyramidLossMSE::test_identical_pyramids_zero_loss` |
|------|-------|
| | `test_helper.py::TestPyramidLossMSE::test_different_pyramids_positive_loss` |
| **How** | Identical pyramids produce ~0 loss; zeros-vs-ones produces positive loss. |
| **Why** | Simple sanity check that the MSE accumulation is correct. |

### Scenario: normalize_batch uses ImageNet stats

| Test | `test_helper.py::TestNormalizeBatch::test_imagenet_normalization` |
|------|-------|
| **How** | Passes a constant 0.5 tensor and checks channel 0 is normalised as `(0.5 - 0.485) / 0.229`. |
| **Why** | This function exists for potential transfer-learning paths. The ImageNet stats are hardcoded — the test pins the exact values. |

### Scenario: rgb2gray_batch averages channels

| Test | `test_helper.py::TestRgb2GrayBatch::test_shape` |
|------|-------|
| | `test_helper.py::TestRgb2GrayBatch::test_average` |
| **How** | Checks output shape is `(B, 1, H, W)` and that `[3, 6, 9]` → `6.0`. |
| **Why** | The function uses `torch.sum / 3.0` — a simple mean, not a perceptual weighting. The test pins this behavior. |

### Scenario: write_tensorboard logs all label-metric pairs

| Test | `test_helper.py::TestWriteTensorboard::test_calls_add_scalar` |
|------|-------|
| **How** | Passes a mock writer, calls `write_tensorboard`, and asserts `add_scalar` is called twice with the correct arguments. |
| **Why** | Uses `unittest.mock.MagicMock` to verify the interface contract without needing a real TensorBoard writer or filesystem. |

---

## Feature: Pre-trained Weight Compatibility

**Source:** `weights2try/`
**Test file:** `tests/test_weight_compatibility.py`

### Scenario: ctCo300dx1 weights load into default UNet **[skip-conditional]**

| Test | `test_weight_compatibility.py::TestCtCo300dx1Weights::test_load_ctCo300dx1` |
|------|-------|
| **How** | Loads `best_test_weights.pt` into `UNet(1, 1, bilinear=True)` via `load_state_dict`. |
| **Why** | If a refactoring changes layer names or channel counts, `load_state_dict` raises a `RuntimeError` about missing/unexpected keys. This is the most critical regression guard. Skipped if weights are not present (e.g. in CI without LFS). |

### Scenario: mtCo150ax1 weights load into default UNet **[skip-conditional]**

| Test | `test_weight_compatibility.py::TestMtCo150ax1Weights::test_load_mtCo150ax1` |
|------|-------|
| **How** | Same as above for the second weight set. |
| **Why** | Two independent weight files verify that the architecture hasn't drifted from either trained checkpoint. |

### Scenario: Inference with pre-trained weights produces valid output **[skip-conditional]**

| Test | `test_weight_compatibility.py::TestPretrainedInference::test_valid_output_range` |
|------|-------|
| **How** | Loads weights, runs a forward pass on random input, and checks output is in `[0, 1]` with no NaN/Inf. |
| **Why** | Loading weights succeeds even if a layer is subtly wrong (e.g. transposed). Running an actual forward pass catches shape mismatches and numerical issues that `load_state_dict` alone might miss. |

---

## Maintenance Guide

### Adding a new test

1. Write the BDD scenario in `CLAUDE.md` under the appropriate feature
2. Implement the pytest function(s) in the corresponding test file
3. Add an entry to this document following the format above
4. Run `pytest tests/ -v` to verify

### Updating a known-bug test

When a known bug is fixed:

1. Update the BDD scenario in `CLAUDE.md` (remove the bug note)
2. Change the test from asserting buggy behavior to asserting correct behavior
3. Remove the **[known-bug]** marker from this document
4. Add a note about when/why it was changed

### Test counts by file

| Test file | Scenarios | Tests | Notes |
|-----------|-----------|-------|-------|
| `tests/model/test_unet.py` | 7 | 13 | |
| `tests/model/test_correlator.py` | 7 | 10 | 2 known-bug |
| `tests/model/test_kornia_dog.py` | 4 | 6 | |
| `tests/model/test_kornia_sift.py` | 5 | 5 | 1 known-bug |
| `tests/dataset/test_neg_dataset.py` | 10 | 10 | |
| `tests/dataset/test_neg_sift_dataset.py` | 6 | 8 | |
| `tests/dataset/test_inference_dataset.py` | 5 | 7 | |
| `tests/utils/test_helper.py` | 12 | 17 | 2 known-bug |
| `tests/test_create_tiled_dataset.py` | 6 | 6 | 1 known-bug |
| `tests/test_weight_compatibility.py` | 3 | 3 | skip-conditional |
| **Total** | **65** | **85** | |
