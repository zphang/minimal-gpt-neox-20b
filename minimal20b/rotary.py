import torch


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer('inv_freq', inv_freq)
        # Delay initialization until first forward call, because initial model on the 'meta' device
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.cos_cached is None:
            t = torch.arange(2048, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # [sx, 1 (b * np), hn]
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


# @torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
