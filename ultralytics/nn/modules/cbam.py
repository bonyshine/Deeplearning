import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
import torch
import torch.nn as nn

# Get the full path to the local ultralytics module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
sys.path.insert(0, project_root)
torch.use_deterministic_algorithms(False)


class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Only avg pool (deterministic)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        attn = self.shared_MLP(avg_pool).unsqueeze(2).unsqueeze(3)
        return self.act(attn)



class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, c1, c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
        
def save_attention_map(attn_map, save_path="cbam_attn.png"):
    plt.figure(figsize=(4, 4))
    plt.imshow(attn_map, cmap='viridis')  # or 'hot', 'plasma'
    plt.axis('off')
    plt.colorbar()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()