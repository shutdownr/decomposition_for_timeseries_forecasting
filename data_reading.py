import numpy as np
import pandas as pd

def read_dataset(dataset):
    data_path = "data/UTS/"

    cif_path = data_path + "cif_2016_dataset.tsf"
    nn5_path = data_path + "nn5_daily_dataset_without_missing_values.tsf"
    tourism_path = data_path + "tourism_monthly_dataset.tsf"
    weather_path = data_path + "weather_prediction_dataset.csv"

    m4_hourly_path = data_path + "Hourly-train.csv"
    m4_weekly_path = data_path + "Weekly-train.csv"
    m4_yearly_path = data_path + "Yearly-train.csv"

    m3_monthly_path = data_path + "M3_monthly_TSTS.csv"
    m3_quarterly_path = data_path + "M3_quarterly_TSTS.csv"
    m3_yearly_path = data_path + "M3_yearly_TSTS.csv"
    m3_other_path = data_path + "M3_other_TSTS.csv"

    transactions_path = data_path + "transactions.csv"
    rain_path = data_path + "mumbai-monthly-rains.csv"

    if dataset == "cif":
        df = read_cif(cif_path)
    elif dataset == "nn5":
        df = read_nn5(nn5_path)
    elif dataset == "tourism":
        df = read_tourism(tourism_path)
    elif dataset == "weather_uts":
        df = read_weather(weather_path)
    elif dataset == "m4_h":
        df = read_m4(m4_hourly_path)
    elif dataset == "m4_w":
        df = read_m4(m4_weekly_path)
    elif dataset == "m4_y":
        df = read_m4(m4_yearly_path)
    elif dataset == "m3_m":
        df = read_m3(m3_monthly_path)
    elif dataset == "m3_q":
        df = read_m3(m3_quarterly_path)
    elif dataset == "m3_y":
        df = read_m3(m3_yearly_path)
    elif dataset == "m3_o":
        df = read_m3(m3_other_path)
    elif dataset == "transactions":
        df = read_transactions(transactions_path)
    elif dataset == "rain":
        df = read_rain(rain_path)
    else:
        return _read_dataset_mts(dataset)
    return df.astype(float)

def read_cif(path):
    df = pd.read_csv(
        path,
        sep=":|,",
        encoding="cp1252",
        engine="python",
        header=None,
        index_col=0,
        skiprows=16
    )
    # Filter for 12 months forecasting horizon
    df = df[df.iloc[:, 0] == 12]
    return df.iloc[:, 1:]

def read_nn5(path):
    df = pd.read_csv(
        path,
        sep=":|,",
        engine="python",
        header=None,
        index_col=0,
        skiprows=19
    )
    return df.iloc[:, 1:]

def read_tourism(path):
    df = pd.read_csv(
        path,
        sep=":",
        encoding="cp1252",
        engine="python",
        header=None,
        index_col=0,
        skiprows=15
    )
    df = df.loc[:, 2].str.split(",", expand=True)
    df = df.astype("float")
    return df

def read_weather(path):
    df = pd.read_csv(
        path,
        sep=","
    )
    columns = df.columns
    temperature_columns = columns.str.endswith("temp_mean")
    df = df.loc[:,temperature_columns]
    df = df.T
    return df

def read_m4(path):
    df = pd.read_csv(path)
    df = df.iloc[:,1:]

    return df

def read_m3(path):
    df = pd.read_csv(path)
    df = pd.DataFrame(df.groupby("series_id")["value"])
    df = df.iloc[:,1]
    all_rows = []
    for row in df:
        all_rows.append(np.array(row))
    
    return pd.DataFrame(all_rows)

def read_transactions(path):
    df = pd.read_csv(path)
    df = pd.pivot(df, index=["store_nbr"], columns=["date"], values=["transactions"])
    df.columns = range(df.columns.size)
    all_rows = []
    for _, row in df.iterrows():
        row.dropna(inplace=True)
        all_rows.append(np.array(row))

    return pd.DataFrame(all_rows)

def read_rain(path):
    df = pd.read_csv(path)
    df = df.iloc[:,1:-1].to_numpy().reshape(-1)
    return pd.DataFrame(df).T

# ------------------------------------------------
# ------------------ MULTIVARIATE ------------------
# ------------------------------------------------

def _read_dataset_mts(dataset):
    data_path = "data/MTS/"

    exchange_rate_path = data_path + "exchange_rate.csv"
    illness_path = data_path + "national_illness.csv"
    weather_path = data_path + "weather.csv"

    ett_h1_path = data_path + "ETTh1.csv"
    ett_h2_path = data_path + "ETTh2.csv"
    ett_m1_path = data_path + "ETTm1.csv"
    ett_m2_path = data_path + "ETTm2.csv"

    covid_path = data_path + "deaths_ages.csv"
    walmart_path = data_path + "walmart_train.csv"

    if dataset == "exchange_rate":
        df = read_exchange_rate(exchange_rate_path)
    elif dataset == "illness":
        df = read_illness(illness_path)
    elif dataset == "weather":
        df = read_weather_mts(weather_path)
    elif dataset == "ett_h1":
        df = read_ett(ett_h1_path)
    elif dataset == "ett_h2":
        df = read_ett(ett_h2_path)
    elif dataset == "ett_m1":
        df = read_ett(ett_m1_path)
    elif dataset == "ett_m2":
        df = read_ett(ett_m2_path)
    elif dataset == "covid":
        df = read_covid(covid_path)
    elif dataset == "walmart":
        df = read_walmart(walmart_path)
    else:
        raise NotImplementedError("Attempting to read unknown dataset")
    
    return df.astype(float)

def read_exchange_rate(path):
    df = pd.read_csv(path).T
    df = df.iloc[1:,:]
    df.reset_index(inplace=True, drop=True)
    df.index = pd.MultiIndex.from_arrays([[0] * len(df), df.index])
    return df

def read_illness(path):
    df = pd.read_csv(path).T
    df = df.iloc[1:,:]
    df.reset_index(inplace=True, drop=True)
    df.index = pd.MultiIndex.from_arrays([[0] * len(df), df.index])
    return df

def read_weather_mts(path):
    df = pd.read_csv(path).T
    df = df.iloc[1:,:]
    df.reset_index(inplace=True, drop=True)
    df.index = pd.MultiIndex.from_arrays([[0] * len(df), df.index])
    return df

def read_ett(path):
    df = pd.read_csv(path).T
    df = df.iloc[1:,:]
    df.reset_index(inplace=True, drop=True)
    df.index = pd.MultiIndex.from_arrays([[0] * len(df), df.index])
    return df

def read_covid(path):
    df = pd.read_csv(path)
    df = df.pivot(columns="Altersgruppe",index="Datum",values="Todesfaelle").T.reset_index().iloc[1:,1:]
    df.index = pd.MultiIndex.from_arrays([[0] * len(df), df.index])
    return df

def read_walmart(path):
    df = pd.read_csv(path)
    df = df.pivot(columns="Date", index=["Store", "Dept"], values="Weekly_Sales")
    df[pd.isna(df)] = 0 # Zero-impute missing data
    all_departments = np.unique(df.index.get_level_values(1))
    all_stores = np.unique(df.index.get_level_values(0))
    for store in all_stores:
        for department in all_departments:
            if (store,department) not in df.index:
                df.loc[(store,department),:] = np.zeros(df.shape[1]) # Zero-impute missing channels
    df.sort_index(inplace=True)
    return df