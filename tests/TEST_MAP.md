# TEST_MAP.md - BDD Scenario to Test Function Mapping

Living document mapping each BDD behavior scenario from CLAUDE.md to its corresponding test function.

---

## Feature: UNet Image Transformation
**Source**: `model/unet.py`, `model/unet_parts.py`
**Test file**: `tests/model/test_unet.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| Forward pass preserves spatial dimensions | `TestUNet::test_forward_pass_preserves_spatial_dims_even` | Passes (1,1,64,64) tensor, asserts output shape matches |
| Forward pass preserves spatial dimensions (odd) | `TestUNet::test_forward_pass_preserves_spatial_dims_odd` | Uses 63x63 input to test skip connection padding |
| Forward pass preserves spatial dimensions (rect) | `TestUNet::test_forward_pass_preserves_spatial_dims_rectangular` | Uses 64x128 to test non-square inputs |
| Output is bounded by sigmoid activation | `TestUNet::test_output_bounded_by_sigmoid` | Feeds large-magnitude input, checks output in [0,1] |
| Bilinear mode uses bicubic upsample with reflect conv | `TestUNet::test_bilinear_mode_uses_bicubic_upsample` | Inspects up1 layer types: nn.Upsample(bicubic) + Conv2d(reflect) |
| Transpose convolution mode uses ConvTranspose2d | `TestUNet::test_transpose_conv_mode` | Creates bilinear=False model, checks all Up blocks use ConvTranspose2d |
| Skip connections handle size mismatches | `TestUNet::test_forward_pass_preserves_spatial_dims_odd` | Odd-sized input forces padding in Up blocks |
| Encoder channel progression [64,128,256,512,512] | `TestUNet::test_encoder_channel_progression` | Runs each encoder stage, checks output channel count |
| DoubleConv applies BatchNorm and ReLU | `TestDoubleConv::test_has_batchnorm_and_relu` | Inspects Sequential children: Conv,BN,ReLU,Conv,BN,ReLU |
| Down block halves spatial dimensions | `TestDown::test_halves_spatial_dimensions` | Passes 32x32 input, asserts 16x16 output |

---

## Feature: Normalized Cross-Correlation
**Source**: `model/correlator.py`
**Test file**: `tests/model/test_correlator.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| Matching embeddings produce high correlation | `TestCorrelator::test_matching_embeddings_high_correlation` | Correlates tensor with its clone, expects > 0.9 |
| Uncorrelated embeddings produce low correlation | `TestCorrelator::test_uncorrelated_embeddings_low_correlation` | Two different random tensors, expects mean |score| < 0.3 |
| Output is scalar per batch element (B,1,1) | `TestCorrelator::test_output_shape` | Checks output.shape == (B, 1, 1) |
| Zero-mean normalization applied | `TestCorrelator::test_zero_mean_normalization` | Calls normalize_batch_zero_mean, checks mean ~ 0 |
| Random noise injected (known bug) | `TestCorrelator::test_random_noise_injection_source_code` | Inspects source for "torch.randn" string |
| Random noise causes nondeterminism | `TestCorrelator::test_random_noise_injection_nondeterminism` | Shows noise causes std values to differ between calls |
| DSIFT normalization flattens H*W*F | `TestCorrelator::test_dsift_normalization` | Calls normalize_batch_zero_mean_dsift, checks shape and mean |
| Correlation normalized by H*W*C | `TestCorrelator::test_correlation_normalized_by_hwc` | Inspects source for "h*w*c" division |

---

## Feature: Difference of Gaussians Pyramid
**Source**: `model/kornia_dog.py`
**Test file**: `tests/model/test_kornia_dog.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| Multi-octave DoG outputs | `TestKorniaDoG::test_multi_octave_output_structure` | Checks return is (list, list, list, list) with non-empty dogs |
| DoG has n_levels-1 layers | `TestKorniaDoG::test_dog_has_n_levels_minus_1_layers` | Sets n_levels=3, checks each DoG has 2 layers at dim=1 |
| Spatial dims decrease across octaves | `TestKorniaDoG::test_spatial_dimensions_decrease_across_octaves` | Compares H across successive octaves |
| KorniaDoGScalePyr variant | `TestKorniaDoGScalePyr::test_scale_pyr_variant_returns_three_elements` | Checks returns 3 elements (no pyramids) |

---

## Feature: SIFT Descriptor Extraction
**Source**: `model/kornia_sift.py`
**Test file**: `tests/model/test_kornia_sift.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| 128-dim descriptors | `TestKorniaSift::test_descriptors_have_128_dimensions` | Runs forward, checks desc.shape[2] == 128 |
| Detection without LAF | `TestKorniaSift::test_detection_without_laf` | Passes laf=None, verifies LAF returned with shape (B,N,2,3) |
| Pre-computed LAF bypass | `TestKorniaSift::test_precomputed_laf_bypass` | Detects LAF, passes it back, verifies same LAF returned |
| Single-channel assertion | `TestKorniaSift::test_single_channel_assertion` | Inspects source for "assert(PC == 1)" |
| 32x32 patch size | `TestKorniaSift::test_patch_size_32` | Checks sift.get_descriptor.patch_size == 32 |

---

## Feature: NCC Siamese Dataset
**Source**: `dataset/neg_dataset.py`
**Test file**: `tests/dataset/test_neg_dataset.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| Paired images from on/off dirs | `TestNegDataset::test_pair_loading_from_on_off_dirs` | Creates matched dirs, verifies len(ds) |
| Missing pair assertion | `TestNegDataset::test_missing_pair_assertion` | Creates orphan on/ file, expects AssertionError |
| ~50% negative sampling | `TestNegDataset::test_negative_sampling_ratio` | Collects all targets, checks ratio between 0.2 and 0.8 |
| Positive-only (weighting=0) | `TestNegDataset::test_positive_only_sampling` | All targets == 1 |
| Negative index differs | `TestNegDataset::test_negative_index_differs` | weighting=1.0, verifies no crash (loop ensures different index) |
| Grayscale conversion | `TestNegDataset::test_grayscale_conversion` | Creates RGB images, checks output has 1 channel |
| Hardcoded normalization | `TestNegDataset::test_hardcoded_normalization` | Checks output values outside [0,1] due to normalization |
| samples_to_use controls size | `TestNegDataset::test_samples_to_use_controls_size` | 20 images * 0.5 = 10 |
| samples_to_use > 1 rejected | `TestNegDataset::test_samples_to_use_greater_than_1_rejected` | Expects AssertionError |
| Output format ((on, off), target) | `TestNegDataset::test_output_format` | Checks tuple structure and types |

---

## Feature: SIFT Siamese Dataset
**Source**: `dataset/neg_sift_dataset.py`
**Test file**: `tests/dataset/test_neg_sift_dataset.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| Consistent augmentation | `TestNegSiftDataset::test_augmentation_applied_consistently` | Checks transform.additional_targets has 'imageOff' |
| Crop dims match original | `TestNegSiftDataset::test_crop_dimensions_match_original` | Verifies output spatial dims == input image size |
| Grayscale OpenCV loading | `TestNegSiftDataset::test_grayscale_opencv_loading` | Loads from RGB PNGs, checks reasonable tensor dims |
| Different normalization stats | `TestNegSiftDataset::test_different_normalization_stats` | Values outside [0,1] due to 0.49/0.135, 0.44/0.12 |
| No negative sampling | `TestNegSiftDataset::test_no_negative_sampling` | Output is (tensor, tensor) not ((tensor, tensor), int) |
| Float32 output | `TestNegSiftDataset::test_float32_output_type` | Checks dtype == torch.float32 |

---

## Feature: Inference Dataset
**Source**: `dataset/inference_dataset.py`
**Test file**: `tests/dataset/test_inference_dataset.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| All PNGs discovered | `TestInferenceDataset::test_directory_png_discovery` | Creates 5 PNGs, checks len(ds) == 5 |
| Non-PNG filtered | `TestInferenceDataset::test_non_png_files_filtered` | Adds .txt and .jpg, checks only PNGs counted |
| Single file path | `TestInferenceDataset::test_single_file_path` | Uses parent dir of single file |
| Configurable normalization | `TestInferenceDataset::test_configurable_normalization` | Custom mean/std shifts values outside [0,1] |
| Basename returned | `TestInferenceDataset::test_filename_returned_as_basename` | Checks no "/" in name, ends with .png |
| Default stats (0.5, 0.1) | `TestInferenceDataset::test_default_normalization_stats` | Inspects __init__ signature defaults |

---

## Feature: Tile Creation
**Source**: `createTiledDataset.py`
**Test file**: `tests/test_create_tiled_dataset.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| Non-overlapping tile count | `TestCreateTiledDataset::test_non_overlapping_tile_count` | 1200x1800 / 600x600 = 6 tiles |
| Overlapping increases count | `TestCreateTiledDataset::test_overlapping_tiles_increase_count` | Compares 0 vs 0.5 overlap tile counts |
| Edge pixels discarded | `TestCreateTiledDataset::test_edge_pixels_discarded` | 700x700 / 600x600 = 1 tile |
| Tile filename format | `TestCreateTiledDataset::test_tile_filename_format` | Checks for name_000000.png pattern |
| Grayscale crash (known bug) | `TestCreateTiledDataset::test_grayscale_crash_known_bug` | 2D array, expects ValueError |
| Output dir must exist | `TestCreateTiledDataset::test_output_dir_must_exist` | Nonexistent dir raises error |

---

## Feature: Pre-trained Weight Compatibility
**Source**: `weights2try/`
**Test file**: `tests/test_weight_compatibility.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| ctCo300dx1 weights load | `TestWeightCompatibility::test_ctco300dx1_weights_load` | load_state_dict succeeds (skip if missing) |
| mtCo150ax1 weights load | `TestWeightCompatibility::test_mtco150ax1_weights_load` | load_state_dict succeeds (skip if missing) |
| Pretrained inference in [0,1] | `TestWeightCompatibility::test_pretrained_inference_output_range` | Forward pass output in [0,1] (skip if missing) |

---

## Feature: Helper Utilities
**Source**: `utils/helper.py`
**Test file**: `tests/utils/test_helper.py`

| BDD Scenario | Test Function | How/Why |
|---|---|---|
| Normer zero-mean output | `TestNormer::test_produces_zero_mean_output` | Checks mean ~ 0 after normalization |
| Normer constant -> NaN (bug) | `TestNormer::test_constant_tensor_produces_nan` | Constant input, asserts NaN/Inf produced |
| Normer epsilon inside std | `TestNormer::test_epsilon_placement_inside_std` | Source inspection for "sample + epsilon" |
| inference_img 2000x2000 cap | `TestInferenceImg::test_caps_at_2000x2000` | Large image, checks output <= 2000 |
| inference_img normalization | `TestInferenceImg::test_normalization` | Checks no NaN, output in [0,1] |
| make_sure_path_exists creates | `TestMakeSurePathExists::test_creates_nested_dirs` | Creates a/b/c, asserts exists |
| make_sure_path_exists idempotent | `TestMakeSurePathExists::test_idempotent` | Calls twice, no error |
| pyramid_loss accumulation | `TestPyramidLoss::test_accumulates_across_levels` | 3-level pyramid, verifies loss >= 0 |
| pyramid_loss channel handling | `TestPyramidLoss::test_channel_handling_reshapes_bc` | Multi-channel pyramid, no crash |
| pyramid_loss_mse | `TestPyramidLoss::test_pyramid_loss_mse` | Random pyramids, loss >= 0 |
| ImageNet normalize_batch | `TestNormalizeBatch::test_imagenet_stats` | Checks channel 0 value matches manual calc |
| rgb2gray_batch | `TestRgb2GrayBatch::test_output_shape` | (B,3,H,W) -> (B,1,H,W) |
| write_tensorboard logs pairs | `TestWriteTensorboard::test_logs_all_pairs` | Mock writer, checks add_scalar call count |
