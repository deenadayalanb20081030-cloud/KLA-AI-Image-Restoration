"""
KLA AI Hackathon: AI-Based Restoration of Degraded Images
Model Architecture: NAFNetSR (Nonlinear Activation Free Network with 2x Super-Resolution Head)

Key Architectural Highlights for NVIDIA H100 Speed & Quality:
1. Nonlinear Activation Free (NAF): Replaces GELU/ReLU with elementwise SimpleGate (x1 * x2),
   reducing FLOPs, increasing GPU arithmetic intensity, and maximizing Tensor Core utilization.
2. Simplified Channel Attention (SCA): Captures global context across feature maps without expensive Softmax.
3. Multi-Scale U-Net Encoder-Decoder: Preserves fine-grained wafer pattern lines & sub-micron edge topology.
4. Lightweight PixelShuffle Upsampling Head: Upscales spatial resolution 2x (128x128 -> 256x256, 256x256 -> 512x512).
5. Dynamic Input Dimension Immunity: Auto-pads to multiples of 8 and unpads at output, handling arbitrary resolutions.
6. Automatic Channel Adaptation: Seamlessly handles 1-channel Grayscale and 3-channel RGB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """
    Nonlinear Activation Free Gate: Splits input along channel dimension
    and performs element-wise multiplication. Replaces GELU/ReLU with zero FLOP overhead.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Core NAFNet Building Block.
    Combines Depthwise Separable Convolutions, SimpleGate, and Simplified Channel Attention.
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * dw_expand
        
        # Spatial Mixing Branch
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)
        
        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True)
        )
        self.sg1 = SimpleGate()
        
        # Feed-Forward Network (FFN) Branch
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)
        self.sg2 = SimpleGate()
        
        # Layer Normalization & Learnable Residual Scaling Parameters
        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Spatial mixing with residual connection
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # Channel mixing (FFN) with residual connection
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNetSR(nn.Module):
    """
    Complete NAFNet Super-Resolution & Joint Denoising Network.
    
    Args:
        img_channels (int): Input image channels (1 for grayscale SEM/optical metrology, 3 for RGB). Default: 1.
        width (int): Number of feature channels in base stage. Default: 32 (ultra-fast) or 64 (high-capacity).
        middle_blk_num (int): Number of NAFBlocks in bottleneck. Default: 8.
        enc_blk_nums (list): Number of NAFBlocks per encoder stage. Default: [2, 2, 4].
        dec_blk_nums (list): Number of NAFBlocks per decoder stage. Default: [2, 2, 2].
        scale (int): Upscaling factor for super-resolution. Default: 2 (2x super-resolution).
    """
    def __init__(
        self,
        img_channels: int = 1,
        width: int = 32,
        middle_blk_num: int = 8,
        enc_blk_nums: list = None,
        dec_blk_nums: list = None,
        scale: int = 2
    ):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2]

        self.in_channels = img_channels
        self.scale = scale
        self.intro = nn.Conv2d(img_channels, width, kernel_size=3, padding=1, stride=1, bias=True)
        
        # Encoder Stages
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        # Bottleneck Stage
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        # Decoder Stages
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan // 2, kernel_size=1, bias=False),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        # Output & 2x PixelShuffle Super-Resolution Head
        self.up_head = nn.Sequential(
            nn.Conv2d(width, width * (scale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            nn.Conv2d(width, img_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel adaptation guard
        if x.shape[1] != self.in_channels:
            if x.shape[1] == 3 and self.in_channels == 1:
                # Convert RGB to Grayscale
                x = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
            elif x.shape[1] == 1 and self.in_channels == 3:
                x = x.repeat(1, 3, 1, 1)

        # Dimension padding guard: Ensure H and W are multiples of 8 for 3-stage U-Net
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_padded = x

        # Base feature extraction
        feat = self.intro(x_padded)
        
        # Encoder forward pass with skip connection caching
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        # Bottleneck processing
        feat = self.middle_blks(feat)

        # Decoder forward pass with skip addition
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = decoder(feat)

        # 2x Super-Resolution reconstruction head
        out = self.up_head(feat)
        
        # Global bicubic residual connection to accelerate convergence & preserve low-frequency background
        bicubic_base = F.interpolate(x_padded, scale_factor=self.scale, mode='bicubic', align_corners=False)
        restored = out + bicubic_base

        # Crop back to exact target dimensions (H*scale, W*scale) if padding was applied
        if pad_h > 0 or pad_w > 0:
            restored = restored[:, :, : h * self.scale, : w * self.scale]

        return restored


def build_model(weights_path: str = None, device: str = 'cuda', width: int = 32) -> NAFNetSR:
    """
    Factory function to instantiate, load weights, and prepare model for inference.
    """
    model = NAFNetSR(img_channels=1, width=width, scale=2)
    if weights_path and torch.cuda.is_available() and weights_path != '':
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"[INFO] Loaded trained weights from: {weights_path}")
        except Exception as e:
            print(f"[WARNING] Could not load weights from {weights_path} ({e}). Initializing random weights.")
            
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    # Unit Test & Dimension Verification
    print("Testing NAFNetSR with standard and non-standard input tensors...")
    net = NAFNetSR(img_channels=1, width=32, scale=2)
    
    # Test 1: 128x128 -> 256x256
    t1 = torch.randn(2, 1, 128, 128)
    out1 = net(t1)
    print(f"Standard Test 1: {t1.shape} -> {out1.shape} (Expected: [2, 1, 256, 256])")
    assert out1.shape == (2, 1, 256, 256), "Dimension mismatch for 128x128 input!"
    
    # Test 2: 256x256 -> 512x512
    t2 = torch.randn(2, 1, 256, 256)
    out2 = net(t2)
    print(f"Standard Test 2: {t2.shape} -> {out2.shape} (Expected: [2, 1, 512, 512])")
    assert out2.shape == (2, 1, 512, 512), "Dimension mismatch for 256x256 input!"

    # Test 3: Arbitrary/Odd Dimension Test (e.g. 127x127 -> 254x254)
    t3 = torch.randn(1, 1, 127, 127)
    out3 = net(t3)
    print(f"Arbitrary Dimension Test: {t3.shape} -> {out3.shape} (Expected: [1, 1, 254, 254])")
    assert out3.shape == (1, 1, 254, 254), "Dimension mismatch for odd dimension input!"
    
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[SUCCESS] Model verified. Trainable Parameters: {params:,} ({params*4/(1024**2):.2f} MB FP32)")
