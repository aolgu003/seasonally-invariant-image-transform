"""Tests for model/unet.py and model/unet_parts.py

Maps to BDD Feature: UNet Image Transformation.
"""

import torch
import torch.nn as nn

from model.unet import UNet
from model.unet_parts import DoubleConv, Down, Up, OutConv


class TestUNet:
    """Tests for the UNet model forward pass and architecture."""

    def test_forward_pass_preserves_spatial_dims_even(self, unet):
        """Scenario: Forward pass preserves spatial dimensions (even input)."""
        x = torch.randn(1, 1, 64, 64)
        out = unet(x)
        assert out.shape == (1, 1, 64, 64)

    def test_forward_pass_preserves_spatial_dims_odd(self, unet):
        """Scenario: Forward pass preserves spatial dimensions (odd input).
        Skip connections handle size mismatches via padding."""
        x = torch.randn(1, 1, 63, 63)
        out = unet(x)
        assert out.shape == (1, 1, 63, 63)

    def test_forward_pass_preserves_spatial_dims_rectangular(self, unet):
        """Scenario: Forward pass preserves spatial dimensions (rectangular)."""
        x = torch.randn(1, 1, 64, 128)
        out = unet(x)
        assert out.shape == (1, 1, 64, 128)

    def test_forward_pass_batch(self, unet):
        """Forward pass works with batch_size > 1."""
        x = torch.randn(2, 1, 64, 64)
        out = unet(x)
        assert out.shape == (2, 1, 64, 64)

    def test_output_bounded_by_sigmoid(self, unet):
        """Scenario: Output is bounded by sigmoid activation [0, 1]."""
        # Use a range of inputs including large values
        x = torch.randn(2, 1, 64, 64) * 10
        out = unet(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_bilinear_mode_uses_bicubic_upsample(self):
        """Scenario: Bilinear mode uses nn.Upsample with bicubic."""
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        # Check up1 block: it should have a Sequential with Upsample(bicubic)
        up_block = model.up1
        assert isinstance(up_block.up, nn.Sequential)
        upsample_layer = up_block.up[0]
        assert isinstance(upsample_layer, nn.Upsample)
        assert upsample_layer.mode == "bicubic"
        assert upsample_layer.scale_factor == 2.0
        # The second layer should be a Conv2d with reflect padding
        conv_layer = up_block.up[1]
        assert isinstance(conv_layer, nn.Conv2d)
        assert conv_layer.padding_mode == "reflect"

    def test_transpose_conv_mode(self):
        """Scenario: Transpose convolution mode uses ConvTranspose2d."""
        model = UNet(n_channels=1, n_classes=1, bilinear=False)
        assert isinstance(model.up1.up, nn.ConvTranspose2d)
        assert isinstance(model.up2.up, nn.ConvTranspose2d)
        assert isinstance(model.up3.up, nn.ConvTranspose2d)
        assert isinstance(model.up4.up, nn.ConvTranspose2d)

    def test_encoder_channel_progression(self, unet):
        """Scenario: Encoder produces expected channel progression [64, 128, 256, 512, 512]."""
        x = torch.randn(1, 1, 64, 64)
        x1 = unet.inc(x)
        assert x1.shape[1] == 64
        x2 = unet.down1(x1)
        assert x2.shape[1] == 128
        x3 = unet.down2(x2)
        assert x3.shape[1] == 256
        x4 = unet.down3(x3)
        assert x4.shape[1] == 512
        x5 = unet.down4(x4)
        assert x5.shape[1] == 512

    def test_multi_channel_input(self):
        """UNet works with n_channels=3 for RGB input."""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_multi_class_output(self):
        """UNet works with n_classes > 1."""
        model = UNet(n_channels=1, n_classes=3, bilinear=True)
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 3, 64, 64)


class TestDoubleConv:
    """Tests for the DoubleConv building block."""

    def test_has_batchnorm_and_relu(self):
        """Scenario: DoubleConv applies BatchNorm and ReLU."""
        block = DoubleConv(1, 64)
        layers = list(block.double_conv.children())
        # Expect: Conv2d, BN, ReLU, Conv2d, BN, ReLU
        assert len(layers) == 6
        assert isinstance(layers[0], nn.Conv2d)
        assert isinstance(layers[1], nn.BatchNorm2d)
        assert isinstance(layers[2], nn.ReLU)
        assert isinstance(layers[3], nn.Conv2d)
        assert isinstance(layers[4], nn.BatchNorm2d)
        assert isinstance(layers[5], nn.ReLU)

    def test_output_channels(self):
        """DoubleConv transforms input channels to output channels."""
        block = DoubleConv(1, 64)
        x = torch.randn(1, 1, 32, 32)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)


class TestDown:
    """Tests for the Down (maxpool + doubleconv) block."""

    def test_halves_spatial_dimensions(self):
        """Scenario: Down block halves spatial dimensions."""
        block = Down(64, 128)
        x = torch.randn(1, 64, 32, 32)
        out = block(x)
        assert out.shape == (1, 128, 16, 16)

    def test_uses_maxpool(self):
        """Down block uses MaxPool2d for downsampling."""
        block = Down(64, 128)
        first_layer = list(block.maxpool_conv.children())[0]
        assert isinstance(first_layer, nn.MaxPool2d)
