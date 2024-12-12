import numpy as np
import pandas as pd

import torch

from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

from decomposition.moving_average import decompose_moving_average, compose_moving_average
from decomposition.trend_seasonality import decompose_trend_seasonality, compose_trend_seasonality
from decomposition.stl import decompose_stl, compose_stl
from decomposition.ssa import decompose_ssa, compose_ssa
from decomposition.fourier_bandlimited import decompose_fourier_bandlimited, compose_fourier_bandlimited
from decomposition.fourier_topk import decompose_fourier_topk, compose_fourier_topk
from decomposition.emd import decompose_emd, compose_emd
from decomposition.wavelet import decompose_wavelet, compose_wavelet
from metrics import mae, mse, rmse, mape, smape, mase, owa, corr

class ForecastingDataset():
    DECOMP_METHODS = [None, "moving_avg", "trend_seasonality", "STL", "SSA", "fourier_bandlimited", "fourier_topk", "wavelet", "EMD"]
    
    TS_VARIABLES = ["ts_train", "ts_full", "X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    _METRICS = {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "smape": smape, "mase": mase, "owa": owa, "corr": corr}
    
    def __init__(self, dataframe, horizon=5, backhorizon=10, train_size=0.6, val_size=0.2, stride_length=1, decomposition=None, decomp_params=None, fixed_origin=False):
        assert train_size + val_size <= 1
        assert decomposition in self.DECOMP_METHODS
        self._horizon = horizon
        self._backhorizon = backhorizon
        self._train_size = train_size
        self._val_size = val_size
        self._stride_length = stride_length
        self._is_scaled = False
        self._fixed_origin = fixed_origin
        self._cut_length = None

        self.is_ragged_length = False
        self.is_multivariate = False
        self.is_ragged = False

        self.process(dataframe)
        if decomposition is not None:
            assert decomp_params is not None
            self.decompose(decomposition, decomp_params)

    def get_attr_flat(self, attr:str, order:str="NCL", make_3d:bool=True):
        order = order.upper()
        assert "N" in order and "L" in order
        if self.is_multivariate:
            assert "C" in order
            default_order = "NCL"
            new_order = [default_order.index(char) for char in order]

        attribute = getattr(self, attr)
        if self.is_ragged or self.is_ragged_length:
            if self.is_multivariate: # Ragged multivariate
                if order == "NCL":
                    out = []
                    for sample in attribute:
                        nr_channels = len(sample)
                        nr_instances = len(sample[0])
                        for instance in range(nr_instances):
                            out_channels = []
                            for channel in range(nr_channels):
                                out_channels.append(sample[channel][instance])
                            out.append(out_channels)
                    if not self.is_ragged_length:
                        return np.array(out)
                    else:
                        return out
                else:
                    raise NotImplementedError("Order not implemented for ragged multivariate data")
            else: # Ragged univariate
                if order == "NL" or order =="NCL":
                    out = []
                    for sample in attribute:
                        if len(sample) > 0:
                            for instance in sample[0]: # Only one channel
                                if make_3d:
                                    out.append([instance])
                                else:
                                    out.append(instance)
                    if not self.is_ragged_length:
                        out = np.array(out)
                    return out
                else: # LN
                    raise NotImplementedError("Order not implemented")
        else:
            if self.is_multivariate: # Non-ragged multivariate
                transposed = attribute.transpose(0,2,1,3)
                transposed = transposed.reshape(-1, transposed.shape[2], transposed.shape[3])
                return transposed.transpose(new_order)
            else: # Non-ragged univariate
                if order == "NL" or order == "NCL":
                    out = attribute.reshape(-1, attribute.shape[-1])
                else:
                    out = attribute.reshape(attribute.shape[-1], -1)
                if make_3d:
                    out = out[:, np.newaxis, :]
                return out
            
    def evaluate(self, y_pred, metrics=["mae","mse"], invert_scaling=True):
        y_pred_ = y_pred.copy()
        y_test_ = self.get_attr_flat("y_test", order="NCL").copy()
        X_test_ = self.get_attr_flat("X_test", order="NCL").copy()
        if invert_scaling:
            assert self._is_scaled
            y_pred_ = self._transform(y_pred.flatten().reshape(-1,1), self._scalers[0], True).reshape(y_pred.shape)
            if self.is_ragged or self.is_ragged_length:
                old_shapes = [len(y_t) for y_t in self.y_test]
                X_test_flattened = np.concatenate(self.X_test).reshape(-1,1)
                y_test_flattened = np.concatenate(self.y_test).reshape(-1,1)
                X_test_transformed = self._transform(X_test_flattened, self._scalers[0], True)
                y_test_transformed = self._transform(y_test_flattened, self._scalers[0], True)
                X_test_ = []
                y_test_ = []
                i = 0
                for shape in old_shapes:
                    X_test_.append(X_test_transformed[i*self._backhorizon:(i+shape)*self._backhorizon].reshape(-1,self._backhorizon))
                    y_test_.append(y_test_transformed[i*self._horizon:(i+shape)*self._horizon].reshape(-1,self._horizon))
                    i += shape
                X_test_ = np.array(X_test_)
                y_test_ = np.array(y_test_)
            else:
                y_test_ = self._transform(self.y_test.flatten().reshape(-1,1), self._scalers[0], True).reshape(self.y_test.shape)
                X_test_ = self._transform(self.X_test.flatten().reshape(-1,1), self._scalers[0], True).reshape(self.X_test.shape)
        
        y_pred_flat = y_pred_
        y_true_flat = y_test_
        X_test_flat = X_test_

        # Flatten everything to 2D to force an ndarray (otherwise ragged ts cause trouble)
        while type(y_pred_flat[0][0]) in [list,np.ndarray]:
            y_pred_flat = [val for list in y_pred_flat for val in list]
        while type(y_true_flat[0][0]) in [list,np.ndarray]:
            y_true_flat = [val for list in y_true_flat for val in list]
        while type(X_test_flat[0][0]) in [list,np.ndarray]:
            X_test_flat = [val for list in X_test_flat for val in list]

        if self.is_ragged_length: # Fix ragged length by just taking the last value (all that matters for MASE calculation)
            X_test_flat = [sample[-2:] for sample in X_test_flat]
        y_pred_flat = np.array(y_pred_flat)
        y_true_flat = np.array(y_true_flat)
        X_test_flat = np.array(X_test_flat)

        errors = {}
        for metric in metrics:
            assert metric in ForecastingDataset._METRICS
            if metric in ["mase", "owa"]:
                error = ForecastingDataset._METRICS[metric](X_test_flat, y_true_flat, y_pred_flat)
            else:
                error = ForecastingDataset._METRICS[metric](y_true_flat, y_pred_flat)
            errors[metric] = error
        return errors

    def _transform(self, data:np.ndarray, scaler, inverse:bool):
        if inverse:
            return scaler.inverse_transform(data)
        else:
            return scaler.transform(data)

    def _scale(self, scaling:str="Standard", by_sample:bool=False, inverse:bool=False):
        if scaling == "Standard":
            Scaler = StandardScaler
        elif scaling == "MinMax":
            Scaler = MinMaxScaler
        elif scaling == "Quantile":
            Scaler = QuantileTransformer
        else:
            raise NotImplementedError("Scaling type not implemented")
        # Read the old ts
        all_ts = self.ts_full
        train_ts = self.ts_train
        nr_samples = len(self.ts_train)
        nr_channels = len(self.ts_train[0])

        self.reset_data()

        # Scale the ts
        if not inverse:
            nr_scalers = nr_samples if by_sample else 1
            self._scalers = np.array([Scaler() for _ in range(nr_scalers)])
            if not by_sample:
                # Reshape for ragged
                channel_out = [[] for _ in range(nr_channels)]
                orig_shapes = []
                for sample in train_ts:
                    for i, channel in enumerate(sample):
                        for instance in channel:
                            channel_out[i].append(instance)
                        orig_shapes.append(len(channel))

                # Fit
                self._scalers[0].fit(np.array(channel_out).T)

        out = []
        for i in range(len(all_ts)):
            scaler_index = i if by_sample else 0
            if not inverse and by_sample:
                # Fit
                self._scalers[scaler_index].fit(train_ts[i].T)
            # Transform
            out.append(self._transform(np.array(all_ts[i]).T, self._scalers[scaler_index], inverse).T)
        all_ts = out

        # Process the scaled ts
        ts_variables = [[] for _ in self.TS_VARIABLES]
        for sample in all_ts:
            # Reset self.TS_VARIABLES
            self.reset_data()
            for channel_ts in sample:
                self.train_test_split(channel_ts)
            if len(getattr(self, self.TS_VARIABLES[0])) == 0:
                continue
            for i, var in enumerate(self.TS_VARIABLES):
                ts_variables[i].append(getattr(self, var))

        for var, ts_variable in zip(self.TS_VARIABLES, ts_variables):
            setattr(self, var, ts_variable)
        
        if self._cut_length:
            self.cut_to_length(self._cut_length)

        self._cast_to_numpy()
        self._is_scaled = not inverse

    def scale(self, scaling="Standard", by_sample=False):
        assert not self._is_scaled
        self._scale_by_sample=by_sample
        return self._scale(scaling, by_sample, False)
        
    def inverse_scale(self):
        assert self._is_scaled
        return self._scale(by_sample=self._scale_by_sample, inverse=True)
    
    def reset_data(self):
        self.TS_VARIABLES = ForecastingDataset.TS_VARIABLES.copy()

        if self._val_size == 0:
            self.TS_VARIABLES = [var for var in self.TS_VARIABLES if "val" not in var]
        self.INSTANCES = [var for var in self.TS_VARIABLES if var.startswith(("X_","y_"))]
        for var in self.TS_VARIABLES: setattr(self, var, [])

    def process(self, dataframe):
        self.reset_data()
        self.is_multivariate = type(dataframe.index) == pd.MultiIndex

        common_length = dataframe.shape[1]
        ts_variables = [[] for _ in self.TS_VARIABLES]
        if self.is_multivariate:
            for _, group in dataframe.groupby(level=0):
                # Reset self.TS_VARIABLES
                self.reset_data()
                for _, ts in group.iterrows():
                    ts = ts.to_numpy()
                    ts = ts[~np.isnan(ts)] # Remove nan values (for ragged datasets)
                    if len(ts) != common_length:
                        self.is_ragged = True
                    self.train_test_split(ts)
                if len(getattr(self, self.TS_VARIABLES[0])) == 0:
                    continue
                for i, var in enumerate(self.TS_VARIABLES):
                    ts_variables[i].append(getattr(self, var))
        else:
            for i in range(len(dataframe)):
                self.reset_data()
                ts = dataframe.iloc[i,:].to_numpy()
                ts = ts[~np.isnan(ts)] # Remove nan values (for ragged datasets)
                if len(ts) != common_length:
                    self.is_ragged = True
                self.train_test_split(ts)
                if len(getattr(self, self.TS_VARIABLES[0])) == 0:
                    continue
                for i, var in enumerate(self.TS_VARIABLES):
                    ts_variables[i].append(getattr(self, var))

        for var, ts_variable in zip(self.TS_VARIABLES, ts_variables):
            setattr(self, var, ts_variable)

        self._cast_to_numpy()
            
    def train_test_split(self, ts, suffix=""):
        def generate_x_y(time_series, fixed_origin=False):
            if self._backhorizon + self._horizon > len(time_series):
                return np.array([], dtype=float), np.array([], dtype=float)
            if fixed_origin:
                indices = np.arange(self._backhorizon,len(time_series)+1-self._horizon)[::self._stride_length]
                x_windows = [time_series[:index] for index in indices]
                y_windows = np.lib.stride_tricks.sliding_window_view(time_series[self._backhorizon:], self._horizon)[::self._stride_length, :]
                return x_windows, y_windows
            else:
                x_windows = np.lib.stride_tricks.sliding_window_view(time_series[:-self._horizon], self._backhorizon)[::self._stride_length, :]
                y_windows = np.lib.stride_tricks.sliding_window_view(time_series[self._backhorizon:], self._horizon)[::self._stride_length, :]
                return x_windows.astype(float), y_windows.astype(float)
        def append_attribute(attr_name, values):
            attr_vals = getattr(self, attr_name)
            attr_vals.append(values)
            setattr(self, attr_name, attr_vals)

        self.is_ragged_length = self._fixed_origin

        train_stop = int((self._train_size + self._val_size) * len(ts))
        ts_train = ts[:train_stop]

        X, y = generate_x_y(ts, self._fixed_origin)

        test_start = int(np.ceil((self._train_size + self._val_size) * len(X)))
        train_stop = int((self._train_size / (self._train_size + self._val_size)) * (test_start - np.ceil(self._horizon / self._stride_length)))
        val_stop = int(test_start - np.ceil(self._horizon / self._stride_length))

        X_train = X[:train_stop]
        y_train = y[:train_stop]
        X_val = X[train_stop:val_stop]
        y_val = y[train_stop:val_stop]
        X_test = X[test_start:]
        y_test = y[test_start:]
        if len(X_test) == 0 or len(X_train) == 0 or (len(X_val) == 0 and self._val_size > 0):
            return
        # Time series (train / full)
        append_attribute("ts_train"+suffix, ts_train)
        append_attribute("ts_full"+suffix, ts)
        # Train instances
        append_attribute("X_train"+suffix, X_train)
        append_attribute("y_train"+suffix, y_train)
        if val_stop > 0: # Val instances
            append_attribute("X_val"+suffix, X_val)
            append_attribute("y_val"+suffix, y_val)
        # Test instances
        append_attribute("X_test"+suffix, X_test)
        append_attribute("y_test"+suffix, y_test)

    def decompose(self, decomp_name, decomp_params):
        if decomp_name == "moving_avg":
            self.decomposition_variables = ["avg", "residual"]
            self.composition_function = compose_moving_average
        elif decomp_name in ["trend_seasonality", "STL"]:
            self.decomposition_variables = ["trend", "seasonal", "residual"]
            self.composition_function = compose_stl if decomp_name == "STL" else compose_trend_seasonality
        elif decomp_name == "SSA":
            self.decomposition_variables = list(np.arange(decomp_params["decomposition_n_subseries"]).astype(str))
            self.composition_function = compose_ssa
        elif decomp_name == "fourier_bandlimited":
            self.decomposition_variables = list(np.arange(decomp_params["decomposition_n_bands"]).astype(str))
            self.composition_function = compose_fourier_bandlimited
        elif decomp_name == "fourier_topk":
            self.decomposition_variables = ["seasonal", "residual"]
            self.composition_function = compose_fourier_topk
        elif decomp_name == "wavelet":
            self.decomposition_variables = ["approximation"] + list(np.arange(decomp_params["decomposition_levels"]).astype(str))
            self.composition_function = compose_wavelet
        elif decomp_name == "EMD":
            self.decomposition_variables = list(np.arange(decomp_params["decomposition_n_subseries"]).astype(str))
            self.composition_function = compose_emd
        elif decomp_name == "none":
            self.decomposition_variables = ["raw"]
            self.composition_function = lambda x: torch.sum(x,axis=0)
        else:
            raise NotImplementedError("Decomposition name not recognized")
        new_instances = self.INSTANCES.copy()
        for variable in ["X_train", "X_val", "X_test"]:
            values = getattr(self, variable)

            out = [[] for _ in range(len(self.decomposition_variables))]
            for o in out:
                for _ in range(len(values)):
                    o.append([])

            for i, item in enumerate(values):
                if decomp_name == "moving_avg":
                    ts_avg, ts_residual = decompose_moving_average(item, kernel_size=decomp_params["decomposition_kernel_size"])
                    out[0][i] = ts_avg
                    out[1][i] = ts_residual
                elif decomp_name in ["trend_seasonality", "STL"]:
                    if decomp_name == "trend_seasonality":
                        ts_trend, ts_seasonal, ts_residual = decompose_trend_seasonality(item, kernel_size=decomp_params["decomposition_kernel_size"], period=decomp_params["decomposition_period"])
                    else:
                        ts_trend, ts_seasonal, ts_residual = decompose_stl(item, period=decomp_params["decomposition_period"])
                    out[0][i] = ts_trend
                    out[1][i] = ts_seasonal
                    out[2][i] = ts_residual
                elif decomp_name == "SSA":
                    for n, subseries in enumerate(decompose_ssa(item, n_subseries=decomp_params["decomposition_n_subseries"], window_size=decomp_params["decomposition_window_size"])):
                        out[n][i] = subseries
                elif decomp_name == "fourier_bandlimited":
                    for n, subseries in enumerate(decompose_fourier_bandlimited(item, n_bands=decomp_params["decomposition_n_bands"])):
                        out[n][i] = subseries
                elif decomp_name == "fourier_topk":
                    ts_seasonal, ts_residual = decompose_fourier_topk(item, k=decomp_params["decomposition_k"])
                    out[0][i] = ts_seasonal
                    out[1][i] = ts_residual
                elif decomp_name == "wavelet":
                    for n, subseries in enumerate(decompose_wavelet(item, wavelet=decomp_params["decomposition_wavelet"], levels=decomp_params["decomposition_levels"])):
                        out[n][i] = subseries
                elif decomp_name == "EMD":
                    for n, subseries in enumerate(decompose_emd(item, n_subseries=decomp_params["decomposition_n_subseries"])):
                        out[n][i] = subseries
                elif decomp_name == "none":
                    out[0][i] = item
                        
            for suffix, data in zip(self.decomposition_variables, out):
                decomposed_name = variable + "_" + suffix
                if self.is_ragged or self.is_ragged_length:
                    setattr(self, decomposed_name, data)
                else:
                    setattr(self, decomposed_name, np.array(data))
                new_instances.append(decomposed_name)
        self.is_decomposed = True
        self.INSTANCES = new_instances

    def cut_to_length(self, length:int):
        if not self._fixed_origin and length > self._backhorizon:
            print("Error: Cannot cut to a length longer than the original backhorizon")
        # NOTE: If the length is greater than the initially provided backhorizon, some instances need to be dropped
        for attr_x, attr_y in zip(["X_train", "X_val", "X_test"], ["y_train", "y_val", "y_test"]):
            if self.is_decomposed:
                decomp_var_names = [attr_x+"_"+var for var in self.decomposition_variables]
            else:
                decomp_var_names = []
            decomp_vars = [getattr(self, var_name) for var_name in decomp_var_names]
            x = getattr(self, attr_x)
            y = getattr(self, attr_y)

            out_x = []
            out_y = []
            out_vars = [[] for var in self.decomposition_variables]
            for i in range(len(x)):
                if len(x[i]) == 0:
                    continue
                out_by_sample_x = []
                out_by_sample_y = []
                out_by_sample_vars = [[] for _ in self.decomposition_variables]
                nr_channels = len(x[i])
                for j in range(nr_channels):
                    nr_instances = len(x[i][j])
                    new_nr_instances = nr_instances
                    for k in range(nr_instances):
                        if len(x[i][j][k]) < length:
                            new_nr_instances -= 1
                            continue 
                        out_by_sample_x.append(x[i][j][k][-length:])
                        out_by_sample_y.append(y[i][j][k])
                        for d in range(len(decomp_vars)):
                            out_by_sample_vars[d].append(decomp_vars[d][i][j][k][-length:])

                if new_nr_instances == 0:
                    continue
                out_by_sample_x = np.array(out_by_sample_x).reshape(nr_channels, new_nr_instances, length)
                out_by_sample_y = np.array(out_by_sample_y).reshape(nr_channels, new_nr_instances, -1)

                out_x.append(out_by_sample_x)
                out_y.append(out_by_sample_y)

                for d in range(len(out_by_sample_vars)):
                    out_vars[d].append(np.array(out_by_sample_vars[d]).reshape(nr_channels, new_nr_instances, length))

            if self.is_ragged:
                setattr(self, attr_x, out_x)
                setattr(self, attr_y, out_y)
                for attr, out in zip(decomp_var_names, out_vars):
                    setattr(self, attr, out)

            else:
                setattr(self, attr_x, np.array(out_x))
                setattr(self, attr_y, np.array(out_y))
                for attr, out in zip(decomp_var_names, out_vars):
                    setattr(self, attr, np.array(out))

        self.is_ragged_length = False
        self._cut_length = length
    
    def _cast_to_numpy(self):
        if not self.is_ragged and not self.is_ragged_length:
            for attr in self.TS_VARIABLES:
                setattr(self, attr, np.array(getattr(self, attr)))
        elif not self.is_ragged_length:
            for attr in self.TS_VARIABLES:
                a = getattr(self, attr)
                out = []
                for sample in a:
                    out.append(np.array(sample))
                setattr(self, attr, out)
        else:
            for attr in ["ts_train", "ts_full"]:
                a = getattr(self, attr)
                out = []
                for sample in a:
                    out.append(np.array(sample))
                setattr(self, attr, out)