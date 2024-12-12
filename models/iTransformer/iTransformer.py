import torch
import torch.nn as nn
import torch.nn.functional as F
from models.iTransformer.Transformer_EncDec import Encoder, EncoderLayer
from models.iTransformer.SelfAttention_Family import FullAttention, AttentionLayer
from models.iTransformer.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_in = configs.enc_in
        self.decomposition_variables = configs.decomposition_variables
        self.composition_function = configs.composition_function

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(-1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, N, _ = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)[:, :N, :]
        # dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, :, 0].unsqueeze(2).repeat(1, 1, self.pred_len))
        dec_out = dec_out + (means[:, :, 0].unsqueeze(2).repeat(1, 1, self.pred_len))
        return dec_out
        # dec_out = dec_out.reshape(dec_out.shape[0], len(self.decomposition_variables), self.enc_in, dec_out.shape[-1])
        # dec_out = dec_out.transpose(0,1)
        # return self.composition_function(dec_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, :, -self.pred_len:]  # [B, L, D]