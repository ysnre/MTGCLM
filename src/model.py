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
    def __init__(self, in_channels: int = 17, num_classes: int = 2, stem: str = "imagenet"):
        """
        Girdi kanal sayısına uyarlanmış ResNet-18.

        stem="imagenet": orijinal ImageNet giriş katmanı (7x7 adım 2 + maxpool). v2'deki RESNET_s1
                         koşusu bununla yapıldı. 33x33 yama ilk iki işlemde 9x9'a iner
                         (33 -> 17 -> 9 -> layer2 5 -> layer3 3 -> layer4 2).
        stem="cifar":    küçük girdiler için giriş katmanı (3x3 adım 1, maxpool yok; He vd. 2016'nın
                         CIFAR deneylerindeki gibi): 33 -> layer1 33 -> 17 -> 9 -> 5. Hesap maliyeti artar.
        İki varyantta parametre adları aynıdır; eski checkpoint'ler stem="imagenet" ile yüklenir.
        """
        super(MTGResNet18, self).__init__()
        if stem not in ("imagenet", "cifar"):
            raise ValueError(f"Bilinmeyen stem '{stem}' (imagenet | cifar)")
        self.stem = stem
        self.resnet = models.resnet18(weights=None)
        
        if stem == "cifar":
            self.resnet.conv1 = nn.Conv2d(in_channels, self.resnet.conv1.out_channels,
                                          kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet.maxpool = nn.Identity()
        else:
            # Modify the first conv layer to accept in_channels (17/22) instead of 3
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
    
    # 3. Test ResNet-18 (ImageNet ve CIFAR giriş katmanı)
    for stem in ("imagenet", "cifar"):
        model_resnet = MTGResNet18(stem=stem)
        output_resnet = model_resnet(dummy_input)
        n_par = sum(p.numel() for p in model_resnet.parameters())
        print(f"ResNet18 Model (stem={stem}, {n_par:,} parametre):")
        print(f"  Output Shape: {output_resnet.shape} (Expected: [8, 2])")
