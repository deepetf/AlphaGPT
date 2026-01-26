import torch
import torch.nn as nn
from .config import ModelConfig
from .ops_registry import OpsRegistry

class AlphaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        
        # 动态构建词表: 特征 + 算子
        self.features_list = ModelConfig.INPUT_FEATURES
        self.ops_list = OpsRegistry.list_ops()
        
        self.vocab = self.features_list + self.ops_list
        self.vocab_size = len(self.vocab)
        
        # Embedding
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, ModelConfig.MAX_FORMULA_LEN + 1, self.d_model))
        
        # Transformer Decoder
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        
        # Output Heads
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head_actor = nn.Linear(self.d_model, self.vocab_size)
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx):
        # idx: [Batch, SeqLen]
        B, T = idx.size()
        
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        
        last_emb = x[:, -1, :]
        logits = self.head_actor(last_emb)
        value = self.head_critic(last_emb)
        
        return logits, value