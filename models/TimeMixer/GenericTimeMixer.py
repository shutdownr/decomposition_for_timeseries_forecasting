import torch
import torch.nn as nn
from models.TimeMixer.Autoformer_EncDec import series_decomp
from models.TimeMixer.Embed import DataEmbedding_wo_pos
from models.TimeMixer.StandardNorm import Normalize

from decomposition.decomposition_pytorch import no_decomposition, moving_average_decomposition, trend_seasonality_decomposition, ssa_decomposition, fourier_bandlimited_decomposition, fourier_topk_decomposition, wavelet_decomposition

def get_top_down(decomposition_variable_name):
    if decomposition_variable_name in ["trend", "avg"]:
        return True
    else:
        return False

class MultiScaleMixing(nn.Module):
    """
    Bottom-up or top-down mixing
    """

    def __init__(self, configs, top_down=False):
        super(MultiScaleMixing, self).__init__()
        self.top_down = top_down
        z1 = 1 if self.top_down else 0
        z2 = 0 if self.top_down else 1
        i_range = range(configs.down_sampling_layers)
        if self.top_down:
            i_range = reversed(i_range)
        self.resampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i+z1)),
                        configs.seq_len // (configs.down_sampling_window ** (i+z2)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i+z2)),
                        configs.seq_len // (configs.down_sampling_window ** (i+z2)),
                    ),
                )
                for i in i_range
            ]
        )

    def forward(self, input_list):
        input_list_ = input_list.copy()
        if self.top_down:
            input_list_.reverse()
        a = input_list_[0]
        b = input_list_[1]
        out_list = [input_list_[0]]
        for i in range(len(input_list_) - 1):
            out_low_res = self.resampling_layers[i](a)
            a = b + out_low_res
            if i + 2 <= len(input_list_) - 1:
                b = input_list_[i + 2]
            out_list.append(a)

        if self.top_down:
            out_list.reverse()
        return out_list

class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        self.decomposition_variables = configs.decomposition_variables

        if configs.decomp_method == "none":
            self.decomposition = no_decomposition()
        elif configs.decomp_method == "moving_avg":
            self.decomposition = moving_average_decomposition(kernel_size=configs.decomposition_kernel_size)
        elif configs.decomp_method == "trend_seasonality":
            self.decomposition = trend_seasonality_decomposition(kernel_size=configs.decomposition_kernel_size, periodicity=configs.decomposition_period) 
        elif configs.decomp_method == "SSA":
            self.decomposition = ssa_decomposition(window_size=configs.decomposition_window_size, n_components=configs.decomposition_n_subseries)
        elif configs.decomp_method == "fourier_bandlimited":
            self.decomposition = fourier_bandlimited_decomposition(n_bands=configs.decomposition_n_bands)
        elif configs.decomp_method == "fourier_topk":
            self.decomposition = fourier_topk_decomposition(top_k=configs.decomposition_k)
        elif configs.decomp_method == "wavelet":
            self.decomposition = wavelet_decomposition(n_levels=configs.decomposition_levels, wavelet=configs.decomposition_wavelet)
        else:
            raise ValueError("decomposition unrecognized")
        self.composition = lambda x: torch.sum(x, dim=0)

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        self.mixing_modules = nn.ModuleList([MultiScaleMixing(configs, get_top_down(var)) for var in self.decomposition_variables])

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        # Decompose to obtain components
        decomposed_lists = [[] for _ in range(len(self.decomposition_variables))]
        for x in x_list:
            components = self.decomposition(x)
            for i in range(len(components)):
                if not self.channel_independence: # Optionally apply the cross layer
                    decomposed_lists[i].append(self.cross_layer(components[i]))
                decomposed_lists[i].append(components[i])
        # Apply multi-scale mixing
        out_lists = [mix_module(component) for mix_module, component in zip(self.mixing_modules, decomposed_lists)]
        # (c, n, N, C, L), where n is ragged

        # Reshape, out is of shape (n,c,N,C,L)
        out = [[] for _ in range(len(x_list))]
        for components in out_lists:
            for j, p in enumerate(components):
                out[j].append(p)
        
        out_recomposed = []
        for o, x in zip(out, x_list):
            # Reconstruct
            final = self.composition(torch.stack(o))
            if self.channel_independence:
                final = x + self.out_cross_layer(final.permute(0,2,1)).permute(0,2,1)
            out_recomposed.append(final[:,:,:x.shape[2]])
        return out_recomposed
            
class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.channel_independence:
            self.projection_layer = nn.Linear(
                configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == "max":
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == "avg":
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode="circular",
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,C,T

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc)
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling)
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::, :self.configs.down_sampling_window])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :, ::self.configs.down_sampling_window]

        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc_sampling_list, x_mark_enc

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out)
                dec_out = self.projection_layer(dec_out.permute(0,2,1)).permute(0,2,1)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).contiguous()
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, N, T = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence:
                    x = x.contiguous().reshape(B * N, 1, T)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, N, T = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence:
                    x = x.contiguous().reshape(B * N, 1, T)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,C,T]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,C,T]
                enc_out_list.append(enc_out)
        # Past Decomposable Mixing as encoder for past        
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out        