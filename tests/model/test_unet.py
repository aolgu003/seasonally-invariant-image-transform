"""Tests for UNet image transformation.

BDD Feature: UNet Image Transformation (model/unet.py, model/unet_parts.py)

Every scenario here locks in the current architectural behavior so that
future refactoring (e.g. parameterising channels for multi-modal support)
cannot silently break shape contracts or activation bounds.
"""

import torch
import torch.nn as nn

from model.unet import UNet
from model.unet_parts import DoubleConv, Down, Up, OutConv


# -- Scenario: Forward pass preserves spatial dimensions -------------------
class TestForwardPassShape:
    """BDD: Given a UNet(1,1) · When a (B,1,H,W) tensor passes through ·
    Then the output shape equals the input shape.

    We check several spatial sizes including an odd dimension to exercise
    the padding logic inside the Up blocks.
    """

    def test_square_even(self, unet):
        x = torch.randn(1, 1, 64, 64)
        assert unet(x).shape == (1, 1, 64, 64)

    def test_square_odd(self, unet):
        x = torch.randn(1, 1, 63, 63)
        assert unet(x).shape == (1, 1, 63, 63)

    def test_rectangular(self, unet):
        x = torch.randn(2, 1, 64, 128)
        assert unet(x).shape == (2, 1, 64, 128)

    def test_batch_dimension_preserved(self, unet):
        x = torch.randn(4, 1, 32, 32)
        assert unet(x).shape[0] == 4


# -- Scenario: Output is bounded by sigmoid activation --------------------
class TestSigmoidOutputRange:
    """BDD: Given any input · When passed through UNet · Then all values
    are in [0, 1].

    The final layer is nn.Sigmoid, so regardless of input distribution
    the output must be bounded.  We test with normal, large-positive, and
    large-negative inputs.
    """

    def test_normal_input(self, unet):
        out = unet(torch.randn(1, 1, 32, 32))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_large_positive_input(self, unet):
        out = unet(torch.ones(1, 1, 32, 32) * 100)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_large_negative_input(self, unet):
        out = unet(torch.ones(1, 1, 32, 32) * -100)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# -- Scenario: Bilinear mode uses bicubic upsample + reflect conv ---------
class TestBilinearMode:
    """BDD: Given UNet(bilinear=True) · Then Up blocks contain
    nn.Upsample(mode='bicubic') and Conv2d(padding_mode='reflect').

    We inspect the module tree rather than running a forward pass because
    we want to verify the *structure*, not just that it runs.
    """

    def test_upsample_is_bicubic(self):
        model = UNet(1, 1, bilinear=True)
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Upsample):
                assert mod.mode == "bicubic"

    def test_reflect_padding_in_up_blocks(self):
        model = UNet(1, 1, bilinear=True)
        up_convs = [
            mod
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Conv2d) and "up" in name
        ]
        reflect_convs = [c for c in up_convs if c.padding_mode == "reflect"]
        assert len(reflect_convs) > 0


# -- Scenario: Transpose convolution mode uses ConvTranspose2d -------------
class TestTransposeMode:
    """BDD: Given UNet(bilinear=False) · Then Up blocks use
    nn.ConvTranspose2d for upsampling.
    """

    def test_has_conv_transpose(self):
        model = UNet(1, 1, bilinear=False)
        transpose_layers = [
            mod
            for mod in model.modules()
            if isinstance(mod, nn.ConvTranspose2d)
        ]
        assert len(transpose_layers) == 4  # one per Up block

    def test_forward_pass_works(self):
        model = UNet(1, 1, bilinear=False)
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        assert model(x).shape == (1, 1, 64, 64)


# -- Scenario: Skip connections handle size mismatches ---------------------
class TestSkipConnectionPadding:
    """BDD: Given odd-sized inputs that cause encoder/decoder dimension
    mismatch · When Up concatenates them · Then padding resolves the
    mismatch.

    The model should not crash on inputs whose spatial dims are not
    powers of two—the Up.forward padding logic handles the remainder.
    """

    def test_odd_input_does_not_crash(self, unet):
        x = torch.randn(1, 1, 47, 53)
        out = unet(x)
        assert out.shape == (1, 1, 47, 53)


# -- Scenario: Encoder produces expected channel progression ---------------
class TestEncoderChannels:
    """BDD: Given a default UNet · Then encoder levels produce channel
    counts [64, 128, 256, 512, 512].

    We verify by inspecting the out_channels of each encoder stage.
    """

    def test_channel_progression(self):
        model = UNet(1, 1, bilinear=True)
        expected = [64, 128, 256, 512, 512]
        stages = [model.inc, model.down1, model.down2, model.down3, model.down4]
        for stage, exp_ch in zip(stages, expected):
            # DoubleConv's last Conv2d tells us the output channels
            convs = [m for m in stage.modules() if isinstance(m, nn.Conv2d)]
            assert convs[-1].out_channels == exp_ch


# -- Scenario: DoubleConv applies BatchNorm and ReLU ----------------------
class TestDoubleConv:
    """BDD: Given a DoubleConv block · When a tensor passes through ·
    Then it traverses Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU.

    We verify by checking the Sequential's children in order.
    """

    def test_layer_order(self):
        block = DoubleConv(1, 64)
        children = list(block.double_conv.children())
        assert isinstance(children[0], nn.Conv2d)
        assert isinstance(children[1], nn.BatchNorm2d)
        assert isinstance(children[2], nn.ReLU)
        assert isinstance(children[3], nn.Conv2d)
        assert isinstance(children[4], nn.BatchNorm2d)
        assert isinstance(children[5], nn.ReLU)

    def test_output_channels(self):
        block = DoubleConv(3, 64)
        x = torch.randn(1, 3, 16, 16)
        assert block(x).shape == (1, 64, 16, 16)


# -- Scenario: Down block halves spatial dimensions ------------------------
class TestDownBlock:
    """BDD: Given a Down block · When (B,C,H,W) passes through ·
    Then output is (B,C_out,H/2,W/2).

    MaxPool2d(2) halves spatial dims, then DoubleConv changes channels.
    """

    def test_spatial_halving(self):
        down = Down(64, 128)
        x = torch.randn(1, 64, 32, 32)
        out = down(x)
        assert out.shape == (1, 128, 16, 16)

    def test_rectangular_halving(self):
        down = Down(64, 128)
        x = torch.randn(2, 64, 48, 64)
        out = down(x)
        assert out.shape == (2, 128, 24, 32)
