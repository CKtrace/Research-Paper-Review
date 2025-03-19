from typing import Dict, List
from torch.nn import ModuleList, Embedding, LayerNorm, Dropout, Softmax, Linear, Module, GELU
from torch import LongTensor, Tensor, einsum, ones, sqrt, tril, triu, cat
from torch.nn.init import normal_, ones_, zeros_ 

class GPT(Module):
    def __init__(self, vocab: int, seq: int, n_layer: int, n_heads: int, dim: int, hiddem: int, dropout: float, device: str):
        super().__init__()
        self.bpe_embed = Embedding(vocab, dim).to(device)
        self.pos_embed = Embedding(seq, dim).to(device)
        self.pos = LongTensor([i for i in range(128)]).to(device)
        self.blocks = ModuleList([
            TransformerBlock(n_heads, dim, hidden, dropout, device) for i in range(n_layers)
        ])
        self.output = Linear(dim, vocab).to(device)
        self.drop = Dropout(dropout).to(device)
        self.init_weights()

    def init_weights(self):
        normal_(self.bpe_embed.weight, mean=0.0, std=0.02)
        normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        normal_(self.output_weight, mean=0.0, std=0.02)
        zeros_(self.output.bias)

    def forward(self, x, ignore):
        be = self.bpe_embed(x)
        pe = self.pos_embed(self.pos)

        out = self.drop(be + pe)
        for block in self.blocks:
            out = block(out, ignore)

        return self.output(out)

    def get_parameters(self) -> List[Dict]:
        params = [
            {'params': [], 'weight_decay': 1e-2},
            {'params': [], 'weight_decay': 0.00}
        ]

        for name, parameter in self.named_parameters():
            if ('att' in name or 'ffl' in name or 'output' in name) and name.endswith('weight'):
                params[0]['params'].append(parameter)
            
            else:
                params[1]['params'].append(parameter)

        return params

class TransformerBlock(Module):

    def __init__(self, n_heads: int, dim: int, hidden: int, dropout: float, device: str):
        super().__init__()
        self.att = MultiHeadAttentionLayer(n_heads, dim, device)
        self.ffl = FeedForwardLayer(dim, hidden, device)
        self.norm1 = LayerNorm(dim).to(device)
        self.norm2 = LayerNorm(dim).to(device)
        self.drop1 = Dropout(dropout).to(device)
        self.drop2 = Dropout(dropout).to(device)
        self.init_weights()

    def init_weights(self):
        ones_(self.norm1.weight)
        ones_(self.norm2.weight)
        zeros_(self.norm1.bias)
        zeros_(self.norm2.bias)

    def forward(self, x, ignore):
        out = self.norm1(x + self.drop1(self.att(x, ignore)))
        return self.norm2(out + self.drop2(self.ffl(out)))

class MultiHeadAttentionLayer(Module):
    def __init__(self, n_heads: int, dim: int, device: str):
        super().__init__()
        self.heads = ModuleList([
            SelfAttention(dim, dim // n_heads, device)
        ])