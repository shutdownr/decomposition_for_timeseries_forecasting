import torch
import torch.nn as nn
from models.NonStationary.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from models.NonStationary.SelfAttention_Family import DSAttention, AttentionLayer
from models.NonStationary.Embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x.permute(0,2,1)).permute(0,2,1)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        self.enc_in = configs.enc_in
        self.decomposition_variables = configs.decomposition_variables
        self.composition_function = configs.composition_function

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in*len(self.decomposition_variables), configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
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
        self.dec_embedding = DataEmbedding(configs.dec_in*len(self.decomposition_variables), configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, len(self.decomposition_variables)*configs.c_out, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.enc_in*len(self.decomposition_variables), seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in*len(self.decomposition_variables), seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(-1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()

        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
                 
        x_dec_new = torch.cat([x_enc[:, :, -self.label_len:], torch.zeros_like(x_enc[:, :, -self.pred_len:])],
                              dim=-1).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out.permute(0,2,1), attn_mask=None, tau=tau, delta=delta)
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out.permute(0,2,1), enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)
        dec_out = dec_out.permute(0,2,1) * std_enc + mean_enc
        # Non-stationary transformer cannot handle some erratic time series
        # In those cases tau is infinite / extremely large, resort to a naive forecast
        if not self.training and torch.isnan(dec_out).sum() > 0:
            return torch.full_like(dec_out, x_enc[0][0][-1])
        
        return dec_out
        # dec_out = dec_out.reshape(dec_out.shape[0], len(self.decomposition_variables), self.enc_in, dec_out.shape[-1])
        # dec_out = dec_out.transpose(0,1)
        # return self.composition_function(dec_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[..., -self.pred_len:]  # [B, D, L]
