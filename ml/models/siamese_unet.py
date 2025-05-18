# ml/models/siamese_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEncoder(nn.Module):
    """Encoder part of the U-Net, used as a shared backbone for Siamese network."""
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(UNetEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Initial block
        self.encoder_blocks.append(self._conv_block(in_channels, features[0]))

        # Subsequent encoder blocks
        for i in range(len(features) - 1):
            self.encoder_blocks.append(self._conv_block(features[i], features[i+1]))

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connections.append(x)
            if i < len(self.encoder_blocks) - 1:
                x = self.pool(x)
        # Return the final feature map and skip connections (reversed for decoder)
        return x, skip_connections[::-1][1:] # Exclude the bottleneck output from skips

class SiameseUNet(nn.Module):
    """Siamese U-Net for Change Detection."""
    def __init__(self, in_channels=4, out_channels=2, features=[64, 128, 256, 512]):
        super(SiameseUNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, features)
        self.bottleneck_channels = features[-1]

        # Decoder (similar to U-Net decoder, but takes concatenated difference)
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Upsampling layers
        for i in range(len(features) - 1, 0, -1):
            self.up_convs.append(
                nn.ConvTranspose2d(features[i], features[i-1], kernel_size=2, stride=2)
            )
            # Decoder block input channels: upsampled_features + skip_connection_diff
            self.decoder_blocks.append(
                self._conv_block(features[i-1] * 2, features[i-1]) # Times 2 because we concat diffs
            )

        # Final convolution layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        # Standard double convolution block for decoder
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        """
        Forward pass for the Siamese network.
        Args:
            x1 (torch.Tensor): Input tensor for the first time point.
            x2 (torch.Tensor): Input tensor for the second time point.
        Returns:
            torch.Tensor: Output segmentation map indicating change.
        """
        # Pass both inputs through the shared encoder
        features1, skips1 = self.encoder(x1)
        features2, skips2 = self.encoder(x2)

        # Start decoding from the bottleneck difference
        x = torch.abs(features1 - features2) # Use absolute difference at bottleneck

        # Decoder loop
        for i in range(len(self.decoder_blocks)):
            # Upsample the current feature map
            x = self.up_convs[i](x)

            # Calculate difference of skip connections
            skip_diff = torch.abs(skips1[i] - skips2[i])

            # Concatenate upsampled features and skip connection difference
            # Ensure dimensions match after upsampling
            if x.shape != skip_diff.shape:
                # Resize x to match skip_diff spatial dimensions
                x = F.interpolate(x, size=skip_diff.shape[2:], mode='bilinear', align_corners=False)

            concat_features = torch.cat((skip_diff, x), dim=1)

            # Pass through the decoder block
            x = self.decoder_blocks[i](concat_features)

        # Final 1x1 convolution
        output = self.final_conv(x)
        return output

# Example Usage (for testing)
if __name__ == '__main__':
    # Example input tensors (Batch Size, Channels, Height, Width)
    img1 = torch.randn(2, 4, 256, 256)
    img2 = torch.randn(2, 4, 256, 256)

    # Initialize the model
    model = SiameseUNet(in_channels=4, out_channels=2) # 2 classes: No Change, Change

    # Forward pass
    output = model(img1, img2)

    # Print output shape
    print("Input shape:", img1.shape)
    print("Output shape:", output.shape) # Should be (Batch Size, Num Classes, Height, Width)

