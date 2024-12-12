import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.composition_function = configs.composition_function
        self.channels = configs.enc_in
        self.decomposition_variables = len(configs.decomposition_variables)

        self.LinearModules = nn.ModuleList()
        for i in range(self.decomposition_variables):
            self.LinearModules.append(
                nn.Linear(self.seq_len, self.pred_len))
            self.LinearModules[i].weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):
        out_stacked = torch.zeros(x.shape[0], x.shape[1], self.pred_len)
        for c in range(self.decomposition_variables):
            out_stacked[:,c*self.channels:(c+1)*self.channels,:] = self.LinearModules[c](x[:,c*self.channels:(c+1)*self.channels,:])
        out_stacked = out_stacked.reshape(x.shape[0],-1,self.channels, self.pred_len)
        out_unstacked = out_stacked.permute(1,0,2,3)
        return self.composition_function(out_unstacked)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.encoder(x_enc)
        return dec_out[:, :, -self.pred_len:]  # [B, C, L]
