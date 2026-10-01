import torch
import torch.nn as nn
import torchvision.models as models

class MTGConvNet(nn.Module):
    def __init__(self, in_channels: int = 17, num_classes: int = 2, conv_blocks: list = [32, 64, 128], dropout_rate: float = 0.2, use_gap: bool = True):
        """
        Modular 2D CNN for patch-based satellite/radar classification.
        
        Args:
            in_channels (int): Number of input channels (default: 17 = 16 MTG + 1 Radar).
            num_classes (int): Number of output classes (default: 2 = Clear vs Cloudy).
            conv_blocks (list): List of filter counts for each convolutional block.
            dropout_rate (float): Dropout probability.
            use_gap (bool): Whether to use Global Average Pooling. If False, flattens spatial features.
        """
        super(MTGConvNet, self).__init__()
        self.use_gap = use_gap
        
        layers = []
        curr_channels = in_channels
        
        for out_channels in conv_blocks:
            # Conv Block: Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
            layers.append(nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout2d(p=dropout_rate))
            curr_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        if self.use_gap:
            # Global Average Pooling (GAP) makes model independent of patch_size
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            fc_in = curr_channels
        else:
            # Dynamically calculate spatial size after max pooling steps
            spatial_size = 33
            for _ in conv_blocks:
                spatial_size = spatial_size // 2
            fc_in = curr_channels * spatial_size * spatial_size
        
        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.use_gap:
            x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MTGResNet18(nn.Module):
    def __init__(self, in_channels: int = 17, num_classes: int = 2):
        """
        17-channel adapted ResNet-18 model for high-capacity satellite/radar data fusion.
        """
        super(MTGResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None)
        
        # Modify the first conv layer to accept in_channels (17) instead of 3
        self.resnet.conv1 = nn.Conv2d(
            in_channels,
            self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Modify the classification head to match num_classes (2)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class MTGUNet(nn.Module):
    def __init__(self, in_channels: int = 17, num_classes: int = 2):
        """
        Fully Convolutional U-Net model for pixel-level cloud segmentation.
        Takes [B, 17, H, W] and outputs [B, 2, H, W] without modifying spatial dimensions.
        """
        super(MTGUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2) # 33 -> 16
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2) # 16 -> 8
        
        # Bottleneck
        self.bottleneck = self._conv_block(64, 128)
        
        # Decoder (Upsampling)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 8 -> 16
        self.dec2 = self._conv_block(128, 64) # 64 (up) + 64 (enc2) = 128
        
        # For the final upsampling, we need an output of 33, but 16*2 = 32. 
        # Output padding of 1 makes it 33.
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, output_padding=1) # 16 -> 33
        self.dec1 = self._conv_block(64, 32) # 32 (up) + 32 (enc1) = 64
        
        # Final output
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 17, 33, 33]
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        d2 = self.up2(b)
        # Skip connection 2
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        # Skip connection 1
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        return out


class MTGConvNet_LateFusion(nn.Module):
    def __init__(self, in_channels: int = 17, scalar_features: int = 3, num_classes: int = 2, conv_blocks: list = [32, 64, 128], dropout_rate: float = 0.2):
        super(MTGConvNet_LateFusion, self).__init__()
        
        layers = []
        curr_channels = in_channels
        
        for out_channels in conv_blocks:
            layers.append(nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout2d(p=dropout_rate))
            curr_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # Spatial Flattening: Calculate size after max pooling steps
        spatial_size = 33
        for _ in conv_blocks:
            spatial_size = spatial_size // 2
        fc_in = curr_channels * spatial_size * spatial_size
        
        # Add scalar features to the fully connected input dimension
        self.classifier = nn.Sequential(
            nn.Linear(fc_in + scalar_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        # Late Fusion: concatenate flattened image features with scalars
        fused = torch.cat([x, scalars], dim=1)
        out = self.classifier(fused)
        return out


class ResNet18_LateFusion(nn.Module):
    def __init__(self, in_channels: int = 17, scalar_features: int = 3, num_classes: int = 2):
        super(ResNet18_LateFusion, self).__init__()
        self.resnet = models.resnet18(weights=None)
        
        self.resnet.conv1 = nn.Conv2d(
            in_channels,
            self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Remove Global Average Pooling and replace with Flatten
        self.resnet.avgpool = nn.Sequential() # Identity
        # Original output before FC is 512 channels. Spatial size after all resnet stages for 33x33 is 2x2.
        # Wait, 33 -> 17 (conv1) -> 9 (pool) -> 9 (layer1) -> 5 (layer2) -> 3 (layer3) -> 2 (layer4)
        # Let's use adaptive pool to 1x1 to be safe if we want standard ResNet behavior, 
        # but instructions say "Replace the final GAP layer with nn.Flatten()".
        # If we just flatten 512x2x2, it's 2048.
        # Let's dynamically compute it or just use AdaptiveAvgPool2d((1,1)) and Flatten? 
        # Instructions: "Replace the final GAP layer with nn.Flatten(), concatenate the 3-scalar vector, and adjust the final fc layer dimension."
        
        # Actually, if we bypass avgpool, we get [B, 512, 2, 2]. Flatten gives 2048.
        self.flatten = nn.Flatten()
        self.fc_in = 512 * 2 * 2 # 2048
        
        self.resnet.fc = nn.Linear(self.fc_in + scalar_features, num_classes)
        
    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Replace GAP with Flatten
        x = self.flatten(x)
        
        # Concatenate scalars
        fused = torch.cat([x, scalars], dim=1)
        out = self.resnet.fc(fused)
        return out


class CloudUNet_Fusion(nn.Module):
    def __init__(self, in_channels: int = 17, scalar_features: int = 3, num_classes: int = 2):
        super(CloudUNet_Fusion, self).__init__()
        
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bottleneck_channels = 128
        self.bottleneck = self._conv_block(64, self.bottleneck_channels)
        
        # Bottleneck Fusion MLP
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.bottleneck_channels)
        )
        
        self.up2 = nn.ConvTranspose2d(self.bottleneck_channels, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, output_padding=1)
        self.dec1 = self._conv_block(64, 32)
        
        # Dual Output Head
        self.mask_out = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        # Bottleneck Fusion
        # Pass scalars through MLP: [Batch, 3] -> [Batch, bottleneck_channels]
        s_emb = self.scalar_mlp(scalars)
        
        # Broadcast spatially: [Batch, C] -> [Batch, C, 1, 1] -> [Batch, C, H, W]
        s_emb = s_emb.unsqueeze(-1).unsqueeze(-1).expand_as(b)
        
        # Add features (can also concatenate, but addition is standard for keeping channels same)
        b = b + s_emb
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Full spatial mask output [Batch, 2, 33, 33]
        mask = self.mask_out(d1)
        
        return mask

if __name__ == "__main__":
    # Quick test of shapes
    print("Testing modular MTGConvNet and MTGResNet18 architectures...")
    
    dummy_input = torch.randn(8, 17, 33, 33) # [Batch, Channels, Height, Width]
    
    # 1. Test standard GAP Shallow CNN
    model_shallow = MTGConvNet(conv_blocks=[32, 64], use_gap=True)
    output_shallow = model_shallow(dummy_input)
    print(f"Shallow Model with GAP (Blocks: [32, 64]):")
    print(f"  Output Shape: {output_shallow.shape} (Expected: [8, 2])")
    
    # 2. Test Spatial Flattening Shallow CNN (No GAP)
    model_flat = MTGConvNet(conv_blocks=[32, 64], use_gap=False)
    output_flat = model_flat(dummy_input)
    print(f"Shallow Model with Flattening (Blocks: [32, 64]):")
    print(f"  Output Shape: {output_flat.shape} (Expected: [8, 2])")
    
    # 3. Test ResNet-18
    model_resnet = MTGResNet18()
    output_resnet = model_resnet(dummy_input)
    print(f"ResNet18 Model:")
    print(f"  Output Shape: {output_resnet.shape} (Expected: [8, 2])")
