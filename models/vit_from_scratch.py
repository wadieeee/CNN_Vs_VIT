import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=192, img_size=32):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)  # (B, num_patches, emb_size)
        return x

class Attention(nn.Module):
    def __init__(self, emb_size=192, num_heads=3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, batch_first=True)
    def forward(self, x):
        x, weights = self.attn(x, x, x, need_weights=True)
        return x, weights

class MLP(nn.Module):
    def __init__(self, emb_size, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, emb_size*mlp_ratio)
        self.fc2 = nn.Linear(emb_size*mlp_ratio, emb_size)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, emb_size=192):
        super().__init__()
        self.attn = Attention(emb_size)
        self.mlp = MLP(emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class DeiTTiny(nn.Module):
    def __init__(self, num_classes=10, emb_size=192, depth=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(emb_size=emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1,1,emb_size))
        self.blocks = nn.ModuleList([TransformerBlock(emb_size) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        attn_maps = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_maps.append(attn)
        x = self.norm(x)
        return self.head(x[:,0]), attn_maps
