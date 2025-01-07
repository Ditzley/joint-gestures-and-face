import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from .adapter import Adapter

class TransformerModel(nn.Module):
    def __init__(self, seq_len, input_dim, output_dim, hidden_dim=512, encoder_depth=4, encoder_heads=8,
                 decoder_depth=4, decoder_heads=8, mlp_ratio=4., norm_layer=nn.LayerNorm, split=True,
                 adapter_reduction_factor=8):
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        self.blocks = nn.ModuleList([
            Block(hidden_dim, encoder_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=nn.GELU)
            for _ in range(encoder_depth)])
        self.norm = norm_layer(hidden_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(hidden_dim, decoder_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=nn.GELU)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(hidden_dim)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim//2, output_dim)
        )

        self.split = split
        if self.split:
            self.reduction_factor = adapter_reduction_factor
            self.num_tokens = 2

            self.encoder_adapters_f1 = nn.ModuleList([
                Adapter(hidden_dim, hidden_dim, reduction_factor=self.reduction_factor, num_tokens=self.num_tokens)
                for _ in range(encoder_depth)
            ])

            self.encoder_adapters_f2 = nn.ModuleList([
                Adapter(hidden_dim, hidden_dim, reduction_factor=self.reduction_factor, num_tokens=self.num_tokens)
                for _ in range(encoder_depth)
            ])

            self.decoder_adapters_f1 = nn.ModuleList([
                Adapter(hidden_dim, hidden_dim, reduction_factor=self.reduction_factor, num_tokens=self.num_tokens)
                for _ in range(decoder_depth)
            ])

            self.decoder_adapters_f2 = nn.ModuleList([
                Adapter(hidden_dim, hidden_dim, reduction_factor=self.reduction_factor, num_tokens=self.num_tokens)
                for _ in range(decoder_depth)
            ])

            self.encoder_adapters_b1 = nn.ModuleList([
                Adapter(hidden_dim, hidden_dim, reduction_factor=self.reduction_factor, num_tokens=self.num_tokens)
                for _ in range(encoder_depth)
            ])

            self.encoder_adapters_b2 = nn.ModuleList([
                Adapter(hidden_dim, hidden_dim, reduction_factor=self.reduction_factor, num_tokens=self.num_tokens)
                for _ in range(encoder_depth)
            ])

            self.decoder_adapters_b1 = nn.ModuleList([
                Adapter(hidden_dim, hidden_dim, reduction_factor=self.reduction_factor, num_tokens=self.num_tokens)
                for _ in range(decoder_depth)
            ])

            self.decoder_adapters_b2 = nn.ModuleList([
                Adapter(hidden_dim, hidden_dim, reduction_factor=self.reduction_factor, num_tokens=self.num_tokens)
                for _ in range(decoder_depth)
            ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        time_emb = time_emb.repeat(1,x.shape[1],1)

        if self.split:
            f, b = torch.chunk(x, 2, dim=-1)
            f_ctx, b_ctx = torch.chunk(context, 2, dim=-1)
            f_ctx = torch.cat([time_emb, f_ctx], dim=-1)
            b_ctx = torch.cat([time_emb, b_ctx], dim=-1)

            f = torch.cat([f, f_ctx], dim=-1)
            b = torch.cat([b, b_ctx], dim=-1)

            f = self.linear(f)
            b = self.linear(b)

            f += self.pos_embedding
            b += self.pos_embedding

            for i, blk in enumerate(self.blocks):
                f_a = self.encoder_adapters_f1[i](f, b)
                b_a = self.encoder_adapters_b1[i](b, f)
                f = f_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f))))
                b = b_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(b))))

                f_a = self.encoder_adapters_f2[i](f, b)
                b_a = self.encoder_adapters_b2[i](b, f)
                f = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f))))
                b = b_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(b))))

            f = self.norm(f)
            b = self.norm(b)

            for i, blk in enumerate(self.decoder_blocks):
                f_a = self.decoder_adapters_f1[i](f, b)
                b_a = self.decoder_adapters_b1[i](b, f)
                f = f_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f))))
                b = b_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(b))))

                f_a = self.decoder_adapters_f2[i](f, b)
                b_a = self.decoder_adapters_b2[i](b, f)
                f = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f))))
                b = b_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(b))))

            f = self.decoder_norm(f)
            b = self.decoder_norm(b)

            f_out = self.out(f)
            b_out = self.out(b)

            return (f_out, b_out)
        else:
            ctx_emb = context
            ctx_emb = torch.cat([time_emb, ctx_emb], dim=-1)
            x = torch.cat([x, ctx_emb], dim=2)
            x = self.linear(x)
            x += self.pos_embedding
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            return self.out(x)
