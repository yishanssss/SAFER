import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import math


class SelfAttnBlock(nn.Module):
    """
    1) Self-Attn + Residual + LayerNorm
    2) FFN + Residual + LayerNorm
    """
    def __init__(self, d_model, num_heads, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)


    def forward(self, x, attn_mask):
        # x shape: [B, T, d_model]
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask) # Self-attention
        x = attn_out          # Residual
        #if x.size(0) % 32 != 0 and x.device == torch.device('cuda:0'):
        #    print("Check Attention:")
        #    print(x[:2])
        x = self.norm1(x)                       # LayerNorm

        ffn_out = self.ffn(x)
        x = x + ffn_out           # Residual
        x = self.norm2(x)                       # LayerNorm
        return x

class CrossAttnBlock(nn.Module):
    """
    Cross-Attn + Residual + LN + FFN
    """
    def __init__(self, d_model, num_heads, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        # query: [B, Tq, d_model]
        # k, v:    [B, Tk, d_model]
        attn_out, _ = self.cross_attn(query, key, value, attn_mask=mask)
        out = query + self.dropout(attn_out)  # residual
        out = self.norm1(out)

        ffn_out = self.ffn(out)
        out = out + self.dropout(ffn_out)     # residual
        out = self.norm2(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.meds_self_attn = SelfAttnBlock(d_model, num_heads, dim_feedforward, dropout)
        self.vitals_self_attn = SelfAttnBlock(d_model, num_heads, dim_feedforward, dropout)
        self.note_self_attn = SelfAttnBlock(d_model, num_heads, dim_feedforward, dropout)
        self.cross_attn_n2v = CrossAttnBlock(d_model, num_heads, dim_feedforward, dropout)
        self.cross_attn_v2n = CrossAttnBlock(d_model, num_heads, dim_feedforward, dropout)
        self.Linear_n = nn.Linear(d_model * 2, d_model)
        self.Linear_v = nn.Linear(d_model * 2, d_model)


    def forward(self, meds, vitals, notes, attn_mask, mask=None):
        # meds: [B, T, d_model]
        # vitals: [B, T, d_model]
        # notes: [B, T, d_model]
        meds = self.meds_self_attn(meds, attn_mask)
        vitals = self.vitals_self_attn(vitals, attn_mask)
        notes = self.note_self_attn(notes, attn_mask)

        # Cross-attention
        notes_v = self.cross_attn_n2v(notes, vitals, vitals, mask)
        vitals_n = self.cross_attn_v2n(vitals, notes, notes, mask)
        notes = torch.cat([notes_v, notes], dim=2)
        vitals = torch.cat([vitals_n, vitals], dim=2)
        notes = self.Linear_n(notes)
        vitals = self.Linear_v(vitals)
        return meds, vitals, notes

class NextMedPredModel(nn.Module):
    def __init__(self, 
                 vitals_feature_dim,    # feature dimension of the vital signs
                 med_emb_dim,         # medicine embedding size
                 notes_emb_dim,       # notes medicine embedding size
                 #demographic_dim,     # dimension of the demographic features
                 d_model,            # embedding size for the transformer
                 num_classes,
                 demo_dim=4,
                 num_heads=8,           # Number of attention heads
                 dim_feedforward=128,   # Feedforward dimension in the transformer
                 dropout=0.2,
                 num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embedding layers
        #self.demographics_encode = nn.Sequential(
        #    nn.Linear(demographic_dim, d_model),
        #    nn.ReLU(),
        #    nn.Dropout(dropout)
        #)
        self.vitals_lin = nn.Sequential(
            nn.Linear(vitals_feature_dim, d_model//2),
            nn.Linear(d_model//2, d_model),
        )
        self.meds_lin = nn.Sequential(
            nn.Linear(med_emb_dim, d_model//2),
            nn.Linear(d_model//2, d_model),
        )
        self.notes_lin = nn.Sequential(
            nn.Linear(notes_emb_dim, d_model*2),
            nn.Linear(d_model*2, d_model),
        )

        self.demo_lin = nn.Sequential(
            nn.Linear(demo_dim, d_model//2),
            nn.Linear(d_model//2, d_model),
        )

        #Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)
        ])

        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=dropout, bidirectional=False)

        self.fuse1 = nn.Linear(d_model * 2, d_model)

        self.fuse2 = nn.Linear(d_model * 2, d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules(): 
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Embedding):
                embedding_std = (2 / (module.num_embeddings + module.embedding_dim))**0.5
                nn.init.normal_(module.weight, mean=0, std=embedding_std)

    def positional_encoding(self, seq_len, emb_size, device):
        pe = torch.zeros(seq_len, emb_size, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # [1, seq_len, emb_size]

    def masked_average(self, x, mask):
        # x: [B, T, d_model]
        # mask: [B, T]
        mask = mask.unsqueeze(-1).expand_as(x)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1)
    
    def subsequent_mask(self, seq_len):
        subsequent_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).bool()
        return subsequent_mask

    def masked_last_hidden(self, x, mask):
        # x: [B, T, d_model]
        # mask: [B, T]
        max_indices = mask.sum(dim=1) - 1
        #print(max_indices)
        return x[torch.arange(x.size(0)), max_indices.to(x.device)]
    
    def padding_mask(self, seq_len, max_len):
        # seq_len: [B]
        # max_len: scalar
        mask = torch.arange(max_len).to(seq_len.device).expand(len(seq_len), max_len) >= seq_len.unsqueeze(1)
        return mask


    def forward(self, vitals, prev_meds, notes, demo, seq_len):
        # demographics: (batch_size, demographic_dim)
        # vitals: (batch_size, seq_len, vitals_feature_dim)
        # prev_meds: (batch_size, seq_len, num_classes)
        # other_meds: (batch_size, seq_len, other_meds_dim)
        # output: (batch_size, num_classes)

        # Embed the medicine info
        batch_size = vitals.size(0)
        max_len = vitals.size(1)
        #mask = self.padding_mask(seq_len, max_len).to(vitals.device)

        #if batch_size % 32 != 0 and notes.device == torch.device('cuda:0'):
        #    print('Before linear layer:')
        #    print(notes[:2])
        demo = self.demo_lin(demo)

        pos_enc = self.positional_encoding(max_len, self.d_model, vitals.device)
        pos_enc = pos_enc.repeat(batch_size, 1, 1)
        meds_attn = self.meds_lin(prev_meds) + pos_enc
        notes_attn = self.notes_lin(notes) + pos_enc
        vitals_attn = self.vitals_lin(vitals) + pos_enc


        #if batch_size % 32 != 0 and notes.device == torch.device('cuda:0'):
        #    print('After linear layer:')
        #    print(notes_attn[:2])

        #print(mask)
        subs_mask = self.subsequent_mask(max_len).to(vitals.device)
        subs_mask = subs_mask.repeat(batch_size*self.num_heads, 1, 1)

        for layer in self.transformer_layers:
            meds_attn, vitals_attn, notes_attn = layer(meds_attn, vitals_attn, notes_attn, attn_mask=subs_mask)

        #meds_attn = meds_attn[:, -1, :].squeeze().view(batch_size, -1)
        vitals_attn = vitals_attn[:, -1, :].squeeze().view(batch_size, -1)
        notes_attn = notes_attn[:, -1, :].squeeze().view(batch_size, -1)


        #if batch_size % 32 != 0 and vitals.device == torch.device('cuda:0'):
        #    print('After transformer layer:')
        #    print(notes_attn[:2])


        # Classifier
        fused = torch.cat([vitals_attn, notes_attn], dim=1)
        fused = self.fuse1(fused)


        fused, demo = fused.view(batch_size, -1), demo.view(batch_size, -1)
        fused_emb = self.fuse2(torch.cat((fused, demo), dim=1))

        #fused = self.masked_average(notes_attn, ~mask)
        #fused = self.masked_last_hidden(notes_attn, ~mask)
        out = self.classifier(fused_emb)
        #if batch_size % 32 != 0 and vitals.device == torch.device('cuda:0'):
        #    print(out[:2])
        return fused_emb, out


class RefinePrediction(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model+num_classes, d_model)
        self.fc2 = nn.Linear(d_model, d_model//2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(d_model//2, d_model//4)
        self.fc4 = nn.Linear(d_model//4, num_classes)

    def forward(self, x, logits):
        #x = torch.cat([x, logits], dim=1)
        #x = self.fc1(x)
        #x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        try:
            diff = F.kl_div(F.log_softmax(x, dim=1), F.softmax(logits, dim=1), reduction='none').sum(dim=1).unsqueeze(1)
        except:
            raise RuntimeError(f'x: {x.shape}, logits: {logits.shape}')
        #diff = F.sigmoid(torch.norm(x - logits, dim=1).unsqueeze(1))
        return x, diff


if __name__ == "__main__":
    # Test the model
    model = NextMedPredModel(vitals_feature_dim=10, num_classes=5, med_emb_dim=5, notes_emb_dim=5, d_model=32)
    vitals = torch.randn(20, 10, 10).to(torch.device('cuda:0'))
    prev_meds = torch.randn(20, 10, 5).to(torch.device('cuda:0'))
    notes_meds = torch.randn(20, 10, 5).to(torch.device('cuda:0'))
    seq_len = torch.randint(8, 10, (20,)).to(torch.device('cuda:0'))
    model.to(torch.device('cuda:0'))
    output = model(vitals, prev_meds, notes_meds, seq_len)
    print(output.shape)

    
