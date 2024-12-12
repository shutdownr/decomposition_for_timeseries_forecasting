import torch
import torch.nn as nn

class input_decomposition_wrapper(nn.Module):
    """
    Input decomposition wrapper
    """

    def __init__(self, model:nn.Module, decomp:nn.Module, interpretable_outputs=False):
        super(input_decomposition_wrapper, self).__init__()
        self.model = model
        self.decomp = decomp
        self.n_components = decomp.n_components
        self.interpretable_outputs = interpretable_outputs

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        B,C,N = x.shape
        decomposed = self.decomp(x) # c,B,C,N
        reshaped = decomposed.permute(1,0,2,3) # B,c,C,N 
        merged = reshaped.reshape(B,-1,N) # B,c*C,N
        forecast = self.model(merged, x_mark_enc, x_dec, x_mark_dec) # B,c*C,N
        unmerged = forecast.reshape(B,self.n_components,C,N) # B,c,C,N
        out = unmerged.permute(1,0,2,3) # c,B,C,N
        if self.interpretable_outputs:
            self.out_interpretable = out.detach().cpu()
        return torch.sum(out, dim=0) # B,C,N For simplicity, we assume purely additive decomposition
    
class input_decomposition_wrapper_no_decomp(nn.Module):
    """
    Input decomposition wrapper without applying decomposition 
    """

    def __init__(self, model:nn.Module, n_components:int, interpretable_outputs=False):
        super(input_decomposition_wrapper_no_decomp, self).__init__()
        self.model = model
        self.interpretable_outputs = interpretable_outputs
        self.n_components = n_components

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        B,cC,_ = x.shape # Data are already decomposed and merged
        c = int(cC / self.n_components)
        forecast = self.model(x, x_mark_enc, x_dec, x_mark_dec) # B,c*C,L
        unmerged = forecast.reshape(B,self.n_components,c,-1) # B,c,C,L
        out = unmerged.permute(1,0,2,3) # c,B,C,L
        if self.interpretable_outputs:
            self.out_interpretable = out.detach().cpu()
        return torch.sum(out, dim=0) # B,C,L For simplicity, we assume purely additive decomposition