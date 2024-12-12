import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Autoformer.Embed import DataEmbedding_wo_pos
from models.Autoformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.Autoformer.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.decomposition_variables = configs.decomposition_variables
        self.composition_function = configs.composition_function

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        self.enc_in = configs.enc_in

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in*len(self.decomposition_variables), configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in*len(self.decomposition_variables), configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out*len(self.decomposition_variables),
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out*len(self.decomposition_variables), bias=True)
        )

    def forecast(self, x_enc):
        # decomp init
        mean = torch.mean(x_enc, dim=-1).unsqueeze(-1).repeat(1, 1, self.pred_len)
        zeros = torch.zeros([x_enc.shape[0], x_enc.shape[1], self.pred_len], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, :, -self.pred_len:], mean], dim=2)
        seasonal_init = torch.cat([seasonal_init[:, :, -self.pred_len:], zeros], dim=2)
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, None)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out.reshape(dec_out.shape[0], len(self.decomposition_variables), self.enc_in, dec_out.shape[-1])
        dec_out = dec_out.transpose(0,1)
        return self.composition_function(dec_out)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[..., -self.pred_len:]  # [B, D, L]