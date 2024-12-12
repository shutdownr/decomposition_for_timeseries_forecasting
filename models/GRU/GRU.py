import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.GRUModules = nn.ModuleList()
            self.LinearModules = nn.ModuleList()
            for _ in range(self.channels):
                self.GRUModules.append(nn.GRU(self.seq_len,configs.d_ff,num_layers=2))
                self.LinearModules.append(nn.Linear(configs.d_ff, self.pred_len))
                self.LinearModules.weight = nn.Parameter(
                    (1 / configs.d_ff) * torch.ones([self.pred_len, configs.d_ff]))
        else:
            self.GRUModules = nn.GRU(self.seq_len,configs.d_ff,num_layers=2)
            self.LinearModules = nn.Linear(configs.d_ff, self.pred_len)
            self.LinearModules.weight = nn.Parameter(
                (1 / configs.d_ff) * torch.ones([self.pred_len, configs.d_ff]))
            
    def encoder(self, x):
        out = torch.zeros(x.shape[0], x.shape[1], self.pred_len)
        if self.individual:
            for c in range(self.channels):
                out[:,c,:] = self.LinearModules[c](self.GRUModules[c](x[:,c,:])[0])
        else:
            out = self.LinearModules(self.GRUModules(x)[0])
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.encoder(x_enc)
        return dec_out[:, :, -self.pred_len:]  # [B, C, L]
