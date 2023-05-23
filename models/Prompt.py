import sys
import torch
import torch.nn.functional as F
import numpy as np

class PromptPool(torch.nn.Module):

    def __init__(self, taskcla, args):
        super(PromptPool,self).__init__()
        self.args = args
        self.prompt_size = (self.args.pool_size, self.args.prompt_emb_dim)
        self.prompt = torch.nn.Parameter(torch.randn(self.prompt_size))
        self.prompt_key = torch.nn.Parameter(torch.randn(self.prompt_size))
        torch.nn.init.uniform_(self.prompt, -1, 1)
        torch.nn.init.uniform_(self.prompt_key, -1, 1)
        return

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed):
        prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed, dim=1) # B, C

        similarity = torch.matmul(x_embed_norm, prompt_norm.t())
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k

        batched_prompt_raw = self.prompt[idx] # B, top_k, C
        batch_size, top_k, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k, c) # B, top_k, C
        x = torch.cat([batched_prompt, x_embed], dim=1)
        return x
