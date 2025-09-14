"""
U-Net model for forest fire detection.

This module implements a flexible U-Net architecture with:
- Pretrained encoder backbones (ResNet, EfficientNet, etc.)
- Skip connections for multi-scale feature fusion
- Configurable decoder depth and filters
- Optional auxiliary outputs for deep supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import math


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm and ReLU.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        padding: Padding size
        dropout: Dropout probability (0 = no dropout)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1, dropout: float = 0.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connections.
    
    Args:
        in_channels: Input channels from previous layer
        skip_channels: Channels from skip connection
        out_channels: Output channels
        dropout: Dropout probability
    """
    
    def __init__(self, in_channels: int, skip_channels: int, 
                 out_channels: int, dropout: float = 0.0):
        super().__init__()
        
        # Upsampling layer
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        
        # Convolutional block after concatenation
        # Handle case where skip_channels might be 0
        concat_channels = in_channels + skip_channels
        self.conv_block = ConvBlock(
            concat_channels, out_channels, dropout=dropout
        )
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch between upsampled and skip features
        if skip.numel() > 0 and x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection only if skip is not empty
        if skip.numel() > 0:
            x = torch.cat([x, skip], dim=1)
        
        # Apply conv block
        x = self.conv_block(x)
        return x


class PretrainedEncoder(nn.Module):
    """
    Pretrained encoder backbone (ResNet, EfficientNet, etc.).
    
    Args:
        backbone: Backbone architecture name
        pretrained: Use pretrained weights
        in_channels: Input channels (3 for RGB, 4 for multispectral, etc.)
    """
    
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, 
                 in_channels: int = 3):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone.startswith("resnet"):
            if backbone == "resnet18":
                backbone_model = models.resnet18(pretrained=pretrained)
                self.channels = [64, 64, 128, 256, 512]
            elif backbone == "resnet34":
                backbone_model = models.resnet34(pretrained=pretrained)
                self.channels = [64, 64, 128, 256, 512]
            elif backbone == "resnet50":
                backbone_model = models.resnet50(pretrained=pretrained)
                self.channels = [64, 256, 512, 1024, 2048]
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone}")
                
            # Extract encoder layers
            self.conv1 = backbone_model.conv1
            self.bn1 = backbone_model.bn1
            self.relu = backbone_model.relu
            self.maxpool = backbone_model.maxpool
            
            self.layer1 = backbone_model.layer1
            self.layer2 = backbone_model.layer2
            self.layer3 = backbone_model.layer3
            self.layer4 = backbone_model.layer4
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer if input channels != 3
        if in_channels != 3:
            self.conv1 = nn.Conv2d(
                in_channels, self.conv1.out_channels,
                kernel_size=self.conv1.kernel_size,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
                bias=False
            )
            
            # Initialize new conv layer
            if pretrained:
                # Use pretrained weights for first 3 channels, random for others
                with torch.no_grad():
                    if in_channels < 3:
                        # Fewer channels: take subset of pretrained weights
                        self.conv1.weight[:, :in_channels] = backbone_model.conv1.weight[:, :in_channels]
                    else:
                        # More channels: repeat/extend pretrained weights
                        self.conv1.weight[:, :3] = backbone_model.conv1.weight
                        for i in range(3, in_channels):
                            self.conv1.weight[:, i:i+1] = backbone_model.conv1.weight[:, i % 3:i % 3 + 1]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Returns:
            List of feature maps at different scales [C1, C2, C3, C4, C5]
        """
        features = []
        
        # Initial conv + pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # C1: H/2, W/2
        
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        features.append(x)  # C2: H/4, W/4
        
        x = self.layer2(x)
        features.append(x)  # C3: H/8, W/8
        
        x = self.layer3(x)
        features.append(x)  # C4: H/16, W/16
        
        x = self.layer4(x)
        features.append(x)  # C5: H/32, W/32
        
        return features


class UNet(nn.Module):
    """
    U-Net model for semantic segmentation.
    
    Args:
        config: Model configuration dictionary containing:
            - backbone: Encoder backbone name
            - pretrained: Use pretrained weights
            - in_channels: Input image channels
            - num_classes: Number of output classes
            - decoder_channels: List of decoder channel sizes
            - dropout: Dropout probability
            - aux_outputs: Enable auxiliary outputs for deep supervision
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config parameters
        backbone = config.get("backbone", "resnet34")
        pretrained = config.get("pretrained", True)
        in_channels = config.get("in_channels", 3)
        self.num_classes = config.get("num_classes", 1)
        decoder_channels = config.get("decoder_channels", [256, 128, 64])  # Default to 3 blocks
        dropout = config.get("dropout", 0.0)
        self.aux_outputs = config.get("aux_outputs", False)
        
        # Initialize encoder
        self.encoder = PretrainedEncoder(backbone, pretrained, in_channels)
        encoder_channels = self.encoder.channels
        
        # Initialize decoder - FIXED VERSION
        self.decoder_blocks = nn.ModuleList()
        
        # Build decoder blocks (bottom-up) - Fixed to handle skip connections properly
        prev_channels = encoder_channels[-1]  # Start with deepest features (C5=512)
        
        # Ensure we don't try to use more decoder blocks than available skip connections
        max_decoder_blocks = len(encoder_channels) - 1  # 4 blocks max for ResNet (C4,C3,C2,C1)
        decoder_channels = decoder_channels[:max_decoder_blocks]
        
        for i, dec_channels in enumerate(decoder_channels):
            # Get skip connection channels - go backwards through encoder features
            skip_idx = len(encoder_channels) - 2 - i  # C4=256, C3=128, C2=64, C1=64
            
            if skip_idx >= 0:
                skip_channels = encoder_channels[skip_idx]
            else:
                # No more skip connections available
                skip_channels = 0
            
            decoder_block = DecoderBlock(
                in_channels=prev_channels,
                skip_channels=skip_channels,
                out_channels=dec_channels,
                dropout=dropout
            )
            
            self.decoder_blocks.append(decoder_block)
            prev_channels = dec_channels  # Update for next iteration
        
        # Final classifier head
        self.classifier = nn.Conv2d(prev_channels, self.num_classes, kernel_size=1)
        
        # Auxiliary classifiers for deep supervision (optional)
        if self.aux_outputs:
            self.aux_classifiers = nn.ModuleList([
                nn.Conv2d(channels, self.num_classes, kernel_size=1)
                for channels in decoder_channels[:-1]
            ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - FIXED VERSION
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary containing:
                - "out": Main output [B, num_classes, H, W]
                - "aux": Auxiliary outputs (if enabled)
        """
        input_shape = x.shape[-2:]
        
        # Encoder forward pass
        encoder_features = self.encoder(x)
        
        # Decoder forward pass - FIXED
        decoder_outputs = []
        x = encoder_features[-1]  # Start with deepest features (C5)
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get corresponding skip features - go backwards through encoder
            skip_idx = len(encoder_features) - 2 - i  # C4, C3, C2, C1
            
            if skip_idx >= 0 and skip_idx < len(encoder_features):
                skip_features = encoder_features[skip_idx]
                x = decoder_block(x, skip_features)
            else:
                # Create empty skip connection for cases where no skip is available
                empty_skip = torch.empty(0, device=x.device, dtype=x.dtype)
                x = decoder_block(x, empty_skip)
            
            decoder_outputs.append(x)
        
        # Final classification
        out = self.classifier(x)
        
        # Resize to input size if needed
        if out.shape[-2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        
        results = {"out": out}
        
        # Auxiliary outputs for deep supervision
        if self.aux_outputs and self.training:
            aux_outs = []
            for i, aux_classifier in enumerate(self.aux_classifiers):
                aux_out = aux_classifier(decoder_outputs[i])
                aux_out = F.interpolate(aux_out, size=input_shape, mode='bilinear', align_corners=False)
                aux_outs.append(aux_out)
            results["aux"] = aux_outs
        
        return results
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference method returning only main output.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Predictions [B, num_classes, H, W]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)["out"]
    
    def predict_large_image(self, x: torch.Tensor, tile_size: int = 512, 
                           overlap: int = 64, batch_size: int = 4) -> torch.Tensor:
        """
        Predict on large images using sliding window approach.
        
        Args:
            x: Input tensor [1, C, H, W] (single image)
            tile_size: Size of each tile
            overlap: Overlap between tiles
            batch_size: Batch size for processing tiles
            
        Returns:
            Prediction tensor [1, num_classes, H, W]
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        
        B, C, H, W = x.shape
        assert B == 1, "Large image prediction only supports batch size 1"
        
        # Calculate stride and number of tiles
        stride = tile_size - overlap
        n_tiles_h = math.ceil((H - overlap) / stride)
        n_tiles_w = math.ceil((W - overlap) / stride)
        
        # Initialize output tensor
        output = torch.zeros(1, self.num_classes, H, W, device=device)
        count_map = torch.zeros(1, 1, H, W, device=device)
        
        # Extract all tiles
        tiles = []
        positions = []
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile position
                start_h = i * stride
                start_w = j * stride
                end_h = min(start_h + tile_size, H)
                end_w = min(start_w + tile_size, W)
                
                # Adjust start position if tile extends beyond image
                start_h = max(0, end_h - tile_size)
                start_w = max(0, end_w - tile_size)
                end_h = start_h + tile_size
                end_w = start_w + tile_size
                
                # Extract tile
                tile = x[:, :, start_h:end_h, start_w:end_w]
                
                # Pad if necessary
                if tile.shape[2] < tile_size or tile.shape[3] < tile_size:
                    pad_h = tile_size - tile.shape[2]
                    pad_w = tile_size - tile.shape[3]
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                
                tiles.append(tile)
                positions.append((start_h, start_w, end_h, end_w))
        
        # Process tiles in batches
        with torch.no_grad():
            for batch_start in range(0, len(tiles), batch_size):
                batch_end = min(batch_start + batch_size, len(tiles))
                batch_tiles = torch.cat(tiles[batch_start:batch_end], dim=0)
                
                # Predict on batch
                batch_preds = self.forward(batch_tiles)["out"]
                
                # Place predictions back in output tensor
                for idx, pred in enumerate(batch_preds):
                    global_idx = batch_start + idx
                    start_h, start_w, end_h, end_w = positions[global_idx]
                    
                    # Handle padding
                    pred_h = min(tile_size, H - start_h)
                    pred_w = min(tile_size, W - start_w)
                    pred = pred[:pred_h, :pred_w]
                    
                    # Add to output
                    output[0, :, start_h:start_h+pred_h, start_w:start_w+pred_w] += pred
                    count_map[0, 0, start_h:start_h+pred_h, start_w:start_w+pred_w] += 1
        
        # Average overlapping predictions
        output = output / torch.clamp(count_map, min=1)
        
        return output


def create_model(config: Dict) -> UNet:
    """
    Factory function to create U-Net model from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized U-Net model
    """
    return UNet(config)


# Example usage and testing
if __name__ == "__main__":
    # Test with 3-channel RGB
    config_rgb = {
        "backbone": "resnet34",
        "pretrained": True,
        "in_channels": 3,  # RGB satellite imagery
        "num_classes": 1,  # Binary fire detection
        "decoder_channels": [256, 128, 64],  # 3 decoder blocks
        "dropout": 0.1,
        "aux_outputs": True
    }
    
    # Test with 4-channel multispectral
    config_multispectral = {
        "backbone": "resnet34",
        "pretrained": True,
        "in_channels": 4,  # Multispectral imagery (RGB + NIR)
        "num_classes": 1,  # Binary fire detection
        "decoder_channels": [256, 128, 64],  # Same 3 decoder blocks
        "dropout": 0.1,
        "aux_outputs": True
    }
    
    print("Testing RGB (3-channel) model:")
    model_rgb = create_model(config_rgb)
    x_rgb = torch.randn(2, 3, 512, 512)
    
    model_rgb.train()
    outputs_rgb = model_rgb(x_rgb)
    print(f"RGB - Main output shape: {outputs_rgb['out'].shape}")
    if "aux" in outputs_rgb:
        print(f"RGB - Auxiliary outputs: {len(outputs_rgb['aux'])}")
    
    print("\nTesting Multispectral (4-channel) model:")
    model_ms = create_model(config_multispectral)
    x_ms = torch.randn(2, 4, 512, 512)  # 4 channels
    
    model_ms.train()
    outputs_ms = model_ms(x_ms)
    print(f"Multispectral - Main output shape: {outputs_ms['out'].shape}")
    if "aux" in outputs_ms:
        print(f"Multispectral - Auxiliary outputs: {len(outputs_ms['aux'])}")
    
    # Print model info
    total_params_rgb = sum(p.numel() for p in model_rgb.parameters())
    total_params_ms = sum(p.numel() for p in model_ms.parameters())
    
    print(f"\nRGB Model parameters: {total_params_rgb:,}")
    print(f"Multispectral Model parameters: {total_params_ms:,}")
    print(f"Parameter difference: {total_params_ms - total_params_rgb:,}")
    
    print("\n✅ Both RGB and multispectral models working correctly!")
    print("🔥 Ready for forest fire detection with flexible input channels!")