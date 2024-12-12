import numpy as np
import torch
from statsmodels.tsa.seasonal import STL

def decompose_stl(input:np.array, period:int):
    if period == 1: # Fix, as period = 1 doesn't work.
        return input, np.zeros_like(input), np.zeros_like(input)
    
    # Input shape: (C, N, L)
    if len(input) > 1: # Multivariate
        return _decompose_multivariate(input, period)
    else: # Univariate
        input = input[0]
        seasonal = period + 1 if period % 2 == 0 else period
        s = int(seasonal*1.5)
        trend = s+1 if s % 2 == 0 else s
        trends = []
        seasonals = []
        residuals = []
        for instance in input:
            stl = STL(instance, seasonal=seasonal,period=period, trend=trend)
            result = stl.fit()
            trends.append(result.trend)
            seasonals.append(result.seasonal)
            residuals.append(result.resid)
        return [trends], [seasonals], [residuals]

def _decompose_multivariate(input:np.array, period:int):
    trends = []
    seasonals = []
    resids = []
    seasonal = period + 1 if period % 2 == 0 else period
    s = int(seasonal*1.5)
    trend = s+1 if s % 2 == 0 else s

    for channel in input:
        trends_channel = []
        seasonals_channel = []
        resids_channel = []
        for instance in channel:
            stl = STL(instance, seasonal=seasonal,period=period, trend=trend)
            result = stl.fit()
            trends_channel.append(result.trend)
            seasonals_channel.append(result.seasonal)
            resids_channel.append(result.resid)
        trends.append(trends_channel)
        seasonals.append(seasonals_channel)
        resids.append(resids_channel)

    return trends, seasonals, resids
    
def compose_stl(inputs):
    return torch.sum(inputs,axis=0)