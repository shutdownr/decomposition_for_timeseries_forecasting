class Config():
    # Algorithm identfiers
    # Experiment 1, 2
    algorithm_names_input_decomp = ["DLinear", "NonStationaryTransformer", "GRU", "TimeMixer", "iTransformer"]
    # Experiment 3
    # algorithm_names_generic_input = ["GenericNBeats"]
    algorithm_names_generic_input = ["GenericLinear", "GenericGRU", "GenericNBeats"]
    # Experiment 4
    algorithm_names_generic_intra = ["GenericTimeMixer"]
    # Experiment 5
    algorithm_names_parameter_ablation = ["DLinear"]
    # Experiment 7
    algorithm_names_long_setup = ["GenericLinear", "GenericNBeats", "GenericGRU", "DLinear", "NonStationaryTransformer", "GRU", "TimeMixer", "iTransformer"]

    # Experiment 6
    interpretability_datasets = ["transactions", "cif", "ett_h1", "illness", "m4_h"]
    interpretability_algorithms = ["iTransformer", "iTransformer", "iTransformer", "TimeMixer", "DLinear"]
    interpretability_samples = [60,800,200,60,0]
    interpretability_channels = [0,0,0,0,0]
    interpretability_decomps = ["STL","moving_avg","fourier_bandlimited","fourier_topk", "wavelet"]

    # Dataset identifiers
    short_datasets = ["cif", "covid", "walmart"]
    long_datasets = ["m4_h", "weather_uts", "transactions", "rain", "m4_y", "ett_h1", "exchange_rate", "illness"]
    dataset_names = ["m4_h", "weather_uts", "transactions", "cif", "rain", "m4_y", "covid", "ett_h1", "exchange_rate", "illness", "walmart"]

    # Decomposition technique identifiers
    # NOTE: Both EMD and SSA are unstable/non-converging, leading to errors and extremely inconsistent results
    # decomp_input_names = ["wavelet"]
    decomp_input_names = ["none", "moving_avg", "trend_seasonality", "STL", "fourier_bandlimited", "fourier_topk", "wavelet"]
    # decomp_intra_names = ["wavelet"]
    decomp_intra_names = ["none", "moving_avg", "trend_seasonality", "fourier_bandlimited", "fourier_topk", "wavelet"]

    default_decomp_params = {
        "decomposition_kernel_size": 5, # moving-average and trend-seasonality
        "decomposition_period": 1, # trend-seasonality and STL  NOTE: This must be set depending on the dataset
        "decomposition_n_subseries": 2, # emd, ssa
        "decomposition_window_size": 11, # ssa
        "decomposition_n_bands": 6, # fourier bandlimited
        "decomposition_k": 5, # fourier topk
        "decomposition_wavelet": "db4", # wavelet
        "decomposition_levels": 5, # wavelet
    }

    default_short_dataset_decomp_params = {
        "decomposition_kernel_size": 33, # moving-average and trend-seasonality
        "decomposition_period": 1, # trend-seasonality and STL  NOTE: This must be set depending on the dataset
        "decomposition_n_subseries": 2, # emd, ssa
        "decomposition_window_size": 5, # ssa
        "decomposition_n_bands": 4, # fourier bandlimited
        "decomposition_k": 4, # fourier topk
        "decomposition_wavelet": "db1", # wavelet
        "decomposition_levels": 4, # wavelet
    }

    ablation_decomp_params = {
        "none": [
            {}
        ],
        "moving_avg": [
            {"decomposition_kernel_size": 5},
            {"decomposition_kernel_size": 11},
            {"decomposition_kernel_size": 17},
            {"decomposition_kernel_size": 25},
            {"decomposition_kernel_size": 33},
        ],
        "fourier_bandlimited": [
            {"decomposition_n_bands": 2},
            {"decomposition_n_bands": 3},
            {"decomposition_n_bands": 4},
            {"decomposition_n_bands": 5},
            {"decomposition_n_bands": 6},
        ],
        "fourier_topk": [
            {"decomposition_k": 1},
            {"decomposition_k": 2},
            {"decomposition_k": 3},
            {"decomposition_k": 4},
            {"decomposition_k": 5},
        ],
        "wavelet": [
            {"decomposition_wavelet": "db1", "decomposition_levels": 1},
            {"decomposition_wavelet": "db1", "decomposition_levels": 2},
            {"decomposition_wavelet": "db1", "decomposition_levels": 3},
            {"decomposition_wavelet": "db1", "decomposition_levels": 4},
            {"decomposition_wavelet": "db1", "decomposition_levels": 5},

            {"decomposition_wavelet": "db2", "decomposition_levels": 1},
            {"decomposition_wavelet": "db2", "decomposition_levels": 2},
            {"decomposition_wavelet": "db2", "decomposition_levels": 3},
            {"decomposition_wavelet": "db2", "decomposition_levels": 4},
            {"decomposition_wavelet": "db2", "decomposition_levels": 5},

            {"decomposition_wavelet": "db3", "decomposition_levels": 1},
            {"decomposition_wavelet": "db3", "decomposition_levels": 2},
            {"decomposition_wavelet": "db3", "decomposition_levels": 3},
            {"decomposition_wavelet": "db3", "decomposition_levels": 4},
            {"decomposition_wavelet": "db3", "decomposition_levels": 5},

            {"decomposition_wavelet": "db4", "decomposition_levels": 1},
            {"decomposition_wavelet": "db4", "decomposition_levels": 2},
            {"decomposition_wavelet": "db4", "decomposition_levels": 3},
            {"decomposition_wavelet": "db4", "decomposition_levels": 4},
            {"decomposition_wavelet": "db4", "decomposition_levels": 5},

            {"decomposition_wavelet": "db5", "decomposition_levels": 1},
            {"decomposition_wavelet": "db5", "decomposition_levels": 2},
            {"decomposition_wavelet": "db5", "decomposition_levels": 3},
            {"decomposition_wavelet": "db5", "decomposition_levels": 4},
            {"decomposition_wavelet": "db5", "decomposition_levels": 5},
        ]     
    }

    # Metrics
    metrics = ["mae", "mse", "smape", "mase", "owa"]

    # Training set size
    train_size = 0.6
    # Validation set size
    val_size = 0.2

    # Input data length
    backhorizon = 48
    short_backhorizon = 12
    # Forecasting horizon
    horizon = 48
    short_horizon = 12

    # For experiment with long horizons
    long_horizon = 720
    long_backhorizon = 720

    periods = {
        "cif": 1,
        "nn5": 7,
        "tourism": 12,
        "weather_uts": 1,
        "m3_m": 12,
        "m3_q": 4,
        "m3_y": 1,
        "m3_o": 1,
        "m4_h": 24,
        "m4_w": 7,
        "m4_y": 1,
        "transactions": 7,
        "rain": 12,
        "weather": 1,
        "exchange_rate": 7,
        "illness": 1,
        "ett_h1": 24, 
        "ett_h2": 24,
        "ett_m1": 12, 
        "ett_m2": 12,
        "covid": 1,
        "walmart": 1,
    }

    # Stride length (sliding window step size) for generating train/val/test instances, varies by dataset
    stride_lengths = {
        "cif": 1, 
        "nn5": 5,
        "tourism": 5,
        "weather_uts": 5,
        "m3_m": 1,
        "m3_q": 1,
        "m3_y": 1,
        "m3_o": 1,
        "m4_h": 5,
        "m4_w": 5,
        "m4_y": 5,
        "transactions": 5,
        "rain": 1,
        "weather": 5,
        "exchange_rate": 1,
        "illness": 1,
        "ett_h1": 5, 
        "ett_h2": 5,
        "ett_m1": 5, 
        "ett_m2": 5,
        "covid": 1,
        "walmart": 1,
    }

    # Deep learning model parameters varying by dataset
    # These parameters are taken as provided in the original authors' codebase and adapted for the other datasets
    # https://github.com/thuml/Time-Series-Library/blob/main/scripts/short_term_forecast/TimesNet_M4.sh
    d_model = {
        "cif": 16,
        "nn5": 16,
        "tourism": 32,
        "weather_uts": 32,
        "m3_m": 32,
        "m3_q": 64,
        "m3_y": 64,
        "m3_o": 32,
        "m4_h": 32,
        "m4_w": 32,
        "m4_y": 64,
        "transactions": 32,
        "rain": 16,
        "weather": 32,
        "exchange_rate": 96,
        "illness": 768,
        "ett_h1": 16, 
        "ett_h2": 32,
        "ett_m1": 64, 
        "ett_m2": 32,
        "covid": 32,
        "walmart": 32,
    }
    d_ff = {
        "cif": 32,
        "nn5": 16,
        "tourism": 32,
        "weather_uts": 32,
        "m3_m": 32,
        "m3_q": 64,
        "m3_y": 64,
        "m3_o": 32,
        "m4_h": 32,
        "m4_w": 32,
        "m4_y": 64,
        "transactions": 32,
        "rain": 16,
        "weather": 32,
        "exchange_rate": 96,
        "illness": 768,
        "ett_h1": 32, 
        "ett_h2": 32,
        "ett_m1": 64, 
        "ett_m2": 32,
        "covid": 32,
        "walmart": 32,
    }