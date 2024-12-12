from forecasting_dataset import ForecastingDataset

class ModelConfig():
    use_gpu = False
    devices = "0,1,2,3"
    gpu = 0
    use_multi_gpu = False

    embed = "timeF"
    freq = "h"
    patience = 3
    learning_rate = 0.001
    train_epochs = 10
    features = "M"
    lradj = "type1"
    moving_avg = 25
    batch_size = 16

    def __init__(self, data:ForecastingDataset, decomposition_parameters=None):
        self.pred_len = data._horizon
        self.label_len = data._horizon
        self.seq_len = data._cut_length if data._cut_length else data._backhorizon
        
        if data.is_multivariate:
            self.enc_in = data.ts_train[0].shape[0]
            self.dec_in = data.ts_train[0].shape[0]
            self.c_out = data.ts_train[0].shape[0]
        else:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1

        self.interpretable_outputs = False

        # Decomp variables
        if hasattr(data, "decomposition_variables"):
            self.decomposition_variables = data.decomposition_variables
            self.composition_function = data.composition_function
            if decomposition_parameters:
                for k,v in decomposition_parameters.items():
                    setattr(self, k, v)
        # Model parameters
        self.d_ff = 100
        self.d_model = 32
        self.dropout = 0.1
        self.n_heads = 8
        self.d_layers = 1
        self.e_layers = 2
        self.factor = 1
        self.activation = "gelu"
        self.p_hidden_dims = [32,32]
        self.p_hidden_layers = 2
        self.down_sampling_window = 2
        self.down_sampling_layers = 2 if self.seq_len > 12 else 1
        self.channel_independence = 1
        self.decomp_method = "moving_avg" # Model-internal decomposition for TimeMixer
        self.moving_avg = 25
        self.use_norm = 1
        self.down_sampling_method = "avg"
