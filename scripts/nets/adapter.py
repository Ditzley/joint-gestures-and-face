import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim, reduction_factor=8, num_tokens=2, conv_groups=1):
        super().__init__()
        self.down_size = input_dim // reduction_factor
        self.down = nn.Conv1d(input_dim, self.down_size, kernel_size=1, groups=conv_groups, bias=False)
        self.up = nn.Conv1d(self.down_size, output_dim, kernel_size=1, groups=conv_groups, bias=False)

        self.activation = nn.ReLU(inplace=True)

        self.latent_tokens = nn.Parameter(torch.rand((num_tokens, input_dim)))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        # do cross attention
        rep_tokens = repeat(self.latent_tokens, 't d -> b t d', b=x.size(0))

        att_y2t = rep_tokens @ y.transpose(-2, -1)
        att_y2t = F.softmax(att_y2t, dim=-1)
        rep_tokens_res = att_y2t @ y

        rep_tokens = rep_tokens + rep_tokens_res

        att_t2x = x @ rep_tokens.transpose(-2, -1)
        att_t2x = F.softmax(att_t2x, dim=-1)
        x_res = att_t2x @ rep_tokens

        x = x + self.gate * x_res.contiguous()

        # bottleneck
        z = self.down(x.permute(0, 2, 1))
        z = self.activation(z)
        output = self.up(z)

        return output.permute(0, 2, 1)
