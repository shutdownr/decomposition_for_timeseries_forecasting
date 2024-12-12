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
        self.decomposition_variables = configs.decomposition_variables
        self.composition_function = configs.composition_function
        self.individual = individual
        self.channels = configs.enc_in
            
        if self.individual:
            self.GRUModules = nn.ModuleDict()
            self.LinearModules = nn.ModuleDict()
            for variable in self.decomposition_variables:
                self.GRUModules[variable] = nn.ModuleList()
                self.LinearModules[variable] = nn.ModuleList()
                for i in range(self.channels):
                    self.GRUModules[variable].append(nn.GRU(self.seq_len,configs.d_ff,num_layers=2))
                    self.LinearModules[variable].append(nn.Linear(configs.d_ff, self.pred_len))
                    self.LinearModules[variable].weight = nn.Parameter(
                        (1 / configs.d_ff) * torch.ones([self.pred_len, configs.d_ff]))
        else:
            self.GRUModules = nn.ModuleDict()
            self.LinearModules = nn.ModuleDict()
            for variable in self.decomposition_variables:
                self.GRUModules[variable] = nn.GRU(self.seq_len,configs.d_ff,num_layers=2)
                self.LinearModules[variable] = nn.Linear(configs.d_ff, self.pred_len)
                self.LinearModules[variable].weight = nn.Parameter(
                    (1 / configs.d_ff) * torch.ones([self.pred_len, configs.d_ff]))

    def encoder(self, x):
        out_total = []
        for i, variable in enumerate(self.decomposition_variables):
            init = x[:,i*self.channels:(i+1)*self.channels,:]
            if self.individual:
                out_sum = torch.zeros([init.shape[0], init.shape[1], self.pred_len],requires_grad=True)
                for c in range(self.channels):
                    out = self.GRUModules[variable](init[:,c,:])
                    out_sum[:,c,:] = self.LinearModules[variable](out)
                out_total.append(out_sum)
            else:
                out = self.GRUModules[variable](init)[0]
                out = self.LinearModules[variable](out)
                out_total.append(out)
        # return torch.stack(out_total).permute(1,0,2,3).reshape(x.shape[0],-1,self.pred_len)
        return self.composition_function(torch.stack(out_total))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.encoder(x_enc)
        return dec_out[:, :, -self.pred_len:]  # [B, C, L]
