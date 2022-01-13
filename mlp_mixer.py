import torch
import torch.nn as nn
from os.path import join as pjoin

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.Dense(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.LayerNorm_0 = nn.LayerNorm(hidden_dim)
        self.token_mixing = MlpBlock(num_tokens, tokens_mlp_dim)
        self.LayerNorm_1 = nn.LayerNorm(hidden_dim)
        self.channel_mixing = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        out = self.LayerNorm_0(x).transpose(1, 2)
        x = x + self.token_mixing(out).transpose(1, 2)
        out = self.LayerNorm_1(x)
        x = x + self.channel_mixing(out)
        return x
    
    def load_from(self, weights, n_block):
        ROOT = f"MixerBlock_{n_block}"
        with torch.no_grad():
            LayerNorm_0_scale = np2th(weights[pjoin(ROOT, 'LayerNorm_0', "scale")]).t()
            LayerNorm_0_bias = np2th(weights[pjoin(ROOT, 'LayerNorm_0', "bias")]).view(-1)
            LayerNorm_1_scale = np2th(weights[pjoin(ROOT, 'LayerNorm_1', "scale")]).t()
            LayerNorm_1_bias = np2th(weights[pjoin(ROOT, 'LayerNorm_1', "bias")]).view(-1)
            
            self.LayerNorm_0.weight.copy_(LayerNorm_0_scale)
            self.LayerNorm_0.bias.copy_(LayerNorm_0_bias)
            self.LayerNorm_1.weight.copy_(LayerNorm_1_scale)
            self.LayerNorm_1.bias.copy_(LayerNorm_1_bias)
            
            
            token_mixing_0_scale = np2th(weights[pjoin(ROOT, 'token_mixing', "Dense_0/kernel")]).t()
            token_mixing_0_bias = np2th(weights[pjoin(ROOT, 'token_mixing', "Dense_0/bias")]).view(-1)
            token_mixing_1_scale = np2th(weights[pjoin(ROOT, 'token_mixing', "Dense_1/kernel")]).t()
            token_mixing_1_bias = np2th(weights[pjoin(ROOT, 'token_mixing', "Dense_1/bias")]).view(-1)
            channel_mixing_0_scale = np2th(weights[pjoin(ROOT, 'channel_mixing', "Dense_0/kernel")]).t()
            channel_mixing_0_bias = np2th(weights[pjoin(ROOT, 'channel_mixing', "Dense_0/bias")]).view(-1)
            channel_mixing_1_scale = np2th(weights[pjoin(ROOT, 'channel_mixing', "Dense_1/kernel")]).t()
            channel_mixing_1_bias = np2th(weights[pjoin(ROOT, 'channel_mixing', "Dense_1/bias")]).view(-1)
            
            if self.token_mixing.Dense[0].weight.shape == token_mixing_0_scale.shape:
                self.token_mixing.Dense[0].weight.copy_(token_mixing_0_scale)
                self.token_mixing.Dense[0].bias.copy_(token_mixing_0_bias)
                self.token_mixing.Dense[2].weight.copy_(token_mixing_1_scale)
                self.token_mixing.Dense[2].bias.copy_(token_mixing_1_bias)
            self.channel_mixing.Dense[0].weight.copy_(channel_mixing_0_scale)
            self.channel_mixing.Dense[0].bias.copy_(channel_mixing_0_bias)
            self.channel_mixing.Dense[2].weight.copy_(channel_mixing_1_scale)
            self.channel_mixing.Dense[2].bias.copy_(channel_mixing_1_bias)


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_size=224):
        super(MlpMixer, self).__init__()
        num_tokens = (image_size // patch_size)**2

        self.stem = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.MixerBlock = nn.Sequential(*[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.MixerBlock(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    
    def load_from(self, weights):
        with torch.no_grad():
            if self.stem.weight.shape == np2th(weights["stem/kernel"],conv=True).shape:
                self.stem.weight.copy_(np2th(weights["stem/kernel"],conv=True))
            self.ln.weight.copy_(np2th(weights["pre_head_layer_norm/scale"]))
            self.ln.bias.copy_(np2th(weights["pre_head_layer_norm/bias"]))
            
            for bname, block in self.MixerBlock.named_children():
                block.load_from(weights, n_block=bname)
