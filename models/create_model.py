from models.Autoformer.Autoformer import Model as Autoformer
from models.DLinear.DLinear import Model as DLinear
from models.DLinear.GenericLinear import Model as GenericLinear
from models.FEDformer.FEDformer import Model as FEDformer
from models.GRU.GRU import Model as GRU
from models.GRU.GenericGRU import Model as GenericGRU
from models.NBEATS.GenericNBEATS import Model as GenericNBeats
from models.NBEATS.NBEATS import Model as NBeats
from models.NonStationary.Nonstationary_Transformer import Model as NonStationaryTransformer
from models.TimeMixer.TimeMixer import Model as TimeMixer
from models.TimeMixer.GenericTimeMixer import Model as GenericTimeMixer
from models.iTransformer.iTransformer import Model as iTransformer
from models.train_model import _acquire_device
from models.input_decomposition_wrapper import input_decomposition_wrapper, input_decomposition_wrapper_no_decomp
from decomposition.decomposition_pytorch import *
import torch.nn as nn

def create_model_with_wrapper(identifier:str, config, with_decomp=True):
    model = create_model(identifier, config)
    if with_decomp:
        if config.decomp_method == "none":
            decomposition = no_decomposition()
        elif config.decomp_method == "moving_avg":
            decomposition = moving_average_decomposition(kernel_size=config.decomposition_kernel_size)
        elif config.decomp_method == "trend_seasonality":
            decomposition = trend_seasonality_decomposition(kernel_size=config.decomposition_kernel_size, periodicity=config.decomposition_period) 
        elif config.decomp_method == "SSA":
            decomposition = ssa_decomposition(window_size=config.decomposition_window_size, n_components=config.decomposition_n_subseries)
        elif config.decomp_method == "fourier_bandlimited":
            decomposition = fourier_bandlimited_decomposition(n_bands=config.decomposition_n_bands)
        elif config.decomp_method == "fourier_topk":
            decomposition = fourier_topk_decomposition(top_k=config.decomposition_k)
        elif config.decomp_method == "wavelet":
            decomposition = wavelet_decomposition(n_levels=config.decomposition_levels, wavelet=config.decomposition_wavelet)
        else:
            raise ValueError("decomposition unrecognized")
        wrapper = input_decomposition_wrapper(model, decomposition, config.interpretable_outputs)
    else:
        wrapper = input_decomposition_wrapper_no_decomp(model, len(config.decomposition_variables), config.interpretable_outputs)
    return wrapper

def create_model(identifier:str, config):
    device = _acquire_device(config.use_gpu, config.gpu, config.use_multi_gpu, config.devices)
    if identifier == "DLinear":
        model = DLinear(config)
    elif identifier == "GenericLinear":
        model = GenericLinear(config)
    elif identifier == "GRU":
        model = GRU(config)
    elif identifier == "GenericGRU":
        model = GenericGRU(config)
    elif identifier == "NBeats":
        model = NBeats(config)
    elif identifier == "GenericNBeats":
        model = GenericNBeats(config)
    elif identifier == "FEDformer":
        model = FEDformer(config)
    elif identifier == "Autoformer":
        model = Autoformer(config)
    elif identifier == "NonStationaryTransformer":
        model = NonStationaryTransformer(config)
    elif identifier == "TimeMixer":
        model = TimeMixer(config)
    elif identifier == "GenericTimeMixer":
        model = GenericTimeMixer(config)
    elif identifier == "iTransformer":
        model = iTransformer(config)
    else:
        raise NotImplementedError("Model identifier unknown")
    
    if config.use_gpu:
        device_ids = config.devices.split(",")
        device_ids = [int(id_) for id_ in device_ids]
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    return model