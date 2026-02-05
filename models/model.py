import torch
import torch.nn as nn
from torch.nn import functional as F

def apply_rotary_emb(x, cos, sin):
    # Standard RoPE application
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    """ Swish-Gated Linear Unit as used in Llama """
    def __init__(self, n_embd, intermediate_size, dropout):
        super().__init__()
        self.w1 = nn.Linear(n_embd, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU calculation: (swish(w1(x)) * w3(x)) * w2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class LuminousBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.2):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head

        # Attention projections
        self.wqkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        
        # Modern Feed-Forward (SwiGLU)
        # Intermediate size is usually 8/3 of n_embd for SwiGLU
        intermediate_size = int(8/3 * n_embd)
        self.ffn = SwiGLU(n_embd, intermediate_size, dropout)

        self.attention_norm = RMSNorm(n_embd)
        self.ffn_norm = RMSNorm(n_embd)
        self.dropout_p = dropout

    def forward(self, x, cos, sin):
        # 1. Self-Attention Block (Pre-Norm)
        r = x
        x = self.attention_norm(x)
        
        B, T, C = x.shape
        q, k, v = self.wqkv(x).split(C, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Flash Attention (Fast & Memory Efficient)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.dropout_p if self.training else 0
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        x = r + self.wo(out)

        # 2. Feed-Forward Block (Pre-Norm)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class LuminousLM(nn.Module):
    def __init__(self, vocab_size, n_embd=448, n_head=7, n_layer=7, 
                 block_size=256, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([
            LuminousBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        self.norm = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)

        self.block_size = block_size
        
        # Weight tying
        self.output.weight = self.token_embedding.weight

        # Pre-compute RoPE frequencies
        head_size = n_embd // n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        t = torch.arange(block_size)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

        self.gradient_checkpointing = False

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        
        # Slice RoPE buffers to current sequence length
        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]
        
        # ⭐ MODIFIED: Use gradient checkpointing if enabled
        if self.gradient_checkpointing and self.training:
            # Gradient checkpointing: saves memory during training
            for layer in self.layers:
                # Need to wrap the layer call to pass cos/sin
                def create_custom_forward(module):
                    def custom_forward(hidden_states):
                        return module(hidden_states, cos, sin)
                    return custom_forward
                
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    use_reentrant=False
                )
        else:
            # Standard forward pass
            for layer in self.layers:
                x = layer(x, cos, sin)
        
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            # ⭐ MODIFIED: Add ignore_index=-100 for instruct tuning
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-100  # Ignores positions where target == -100
            )
        
        return logits, loss
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
