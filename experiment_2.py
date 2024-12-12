import pickle
import os

import numpy as np

from config import Config
from data_reading import read_dataset
from forecasting_dataset import ForecastingDataset
from model_config import ModelConfig
from models.create_model import create_model_with_wrapper
from models.train_model import train_model, predict
from utility import set_random_seed, components_to_features

RESULTS_FOLDER = "results/experiment_2"

def save_errors(algorithm_name:str, dataset_name:str, decomp_name:str, errors:dict):
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    algo_folder = RESULTS_FOLDER + "/" + algorithm_name
    if not os.path.exists(algo_folder):
        os.makedirs(algo_folder)
    file_name = algo_folder + "/" + dataset_name + ".pkl"
    if os.path.exists(file_name):
        with open(file_name, "rb") as file:
            all_errors = pickle.load(file)
            all_errors[decomp_name] = errors
    else:
        all_errors = {
            decomp_name: errors
        }
    with open(file_name, "wb") as file:
        pickle.dump(all_errors, file)

def main():
    c = Config()
    n_random_states = 3
    
    for dataset in c.dataset_names:
        print("Starting dataset", dataset)
        df = read_dataset(dataset)
        if df.shape[1] < 3 * (c.horizon + c.backhorizon):
            horizon = c.short_horizon
            backhorizon = c.short_backhorizon
            decomp_params = c.default_short_dataset_decomp_params
        else:
            horizon = c.horizon
            backhorizon = c.backhorizon
            decomp_params = c.default_decomp_params
        for decomp in c.decomp_input_names:
            print("Starting decomposition", decomp)
            data = ForecastingDataset(df, 
                                        horizon=horizon, 
                                        backhorizon=backhorizon, 
                                        train_size=c.train_size, 
                                        val_size=c.val_size, 
                                        stride_length=c.stride_lengths[dataset],
                                        decomposition=None,
                                        fixed_origin=True)
            data.scale()

            decomp_params["decomposition_period"] = c.periods[dataset]
            data.decompose(decomp, decomp_params)

            if decomp in ["trend_seasonality", "STL"] and decomp_params["decomposition_period"] == 1:
                print("period == 1", "skipping")
                continue

            data.cut_to_length(c.backhorizon)

            for algorithm in c.algorithm_names_input_decomp:
                print("\n\nStarting algorithm", algorithm, "\n\n")

                total_errors = np.zeros((n_random_states, len(c.metrics)))
                for random_state in np.arange(n_random_states):
                    set_random_seed(random_state)
                    mc = ModelConfig(data, decomposition_parameters=decomp_params)
                    mc.decomp_method = decomp

                    model = create_model_with_wrapper(algorithm, mc, with_decomp=False)
                    model_trained = train_model(model, 
                                                mc, 
                                                components_to_features(data, "X_train"),
                                                data.get_attr_flat("y_train", order="NCL"),
                                                components_to_features(data, "X_val"),
                                                data.get_attr_flat("y_val", order="NCL"),
                                                experiment_path="experiment_2")
                    y_pred = predict(model_trained, mc, components_to_features(data, "X_test"), is_ragged_length=data.is_ragged_length)
                    y_pred = y_pred.squeeze() # Squeeze channels for univariate data

                    errors = data.evaluate(y_pred, metrics=c.metrics, invert_scaling=False)
                    total_errors[random_state,:] = list(errors.values())
                print("Done with algorithm", algorithm)
                mean_errors = np.mean(total_errors,axis=0)
                std_errors = np.std(total_errors,axis=0)
                print(decomp, mean_errors)
                error_dict = dict(zip(c.metrics, mean_errors))
                for metric, error in zip(c.metrics, std_errors):
                    error_dict["std_"+metric] = error
                save_errors(algorithm, dataset, decomp, error_dict)
            print("Done with dataset", dataset)
            print()
        print("\n\n#####################\n\n")

if __name__ == "__main__":
    main()