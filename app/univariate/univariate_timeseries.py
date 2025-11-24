
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Literal, Any, Callable

import utils.utils as utils
import utils.plot_utils as plu
import utils.timeseries_utils as tsu
import data_handling as dh



def univariate_timeseries_analysis(params:dict, plot_limit:int=-1, color:str="blue") -> None:
    '''
    Performs the timeseries analysis and model fit on the data available
    '''
    verbose:bool = params['verbose']

    features = params['output_features']
    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION (TIMESERIES)", highlight=color)
        print("Features:")
        utils.print_two_column(features, color=color)
    
    # RAW DATA
    DF = dh.load_dataset(f"{params['dataset_folder']}/{params['dataset']}",
                         features=set(features),
                         verbose=verbose,
                         normalize=True,
                         reduce_frequency=params['reduce_frequency'],
                        )
    # STATIONARITY
    if params['enforce_stationarity']:
        TS = tsu.stationarity_analysis(DF=DF,
                                   max_diff=None,
                                   verbose=verbose,
                                   color=color,
                                   plot_folder=params['timeseries_folder'],
                                   plot_limit=plot_limit,
                                  )
    else:
        TS = DF.to_numpy()
    
    # ARIMA Model
    assert len(features) == 1
    # train_test_ARIMA(params, TS, verbose, color, plot_limit=plot_limit)
    
    # tsu.plot_ACF_PACF(timeseries=TS,
    #                   output_folder=params['timeseries_folder'],
    #                   series_name=features[0],
    #                   max_lag=params['seq_len'],
    #                   verbose=verbose
    #                  )
    # train_test_AR(params, TS, verbose, color, plot_limit=plot_limit)
    # train_test_MA(params, TS, verbose, color, plot_limit=plot_limit)
    train_test_ARCH(params, TS, verbose, color, plot_limit=plot_limit)
    train_test_GARCH(params, TS, verbose, color, plot_limit=plot_limit)

    
# # # # # # # # # #
# MODEL FUNCTIONS #
# # # # # # # # # #



def train_test_ARIMA(params, TS, verbose:bool=True, color:str="blue", plot_limit:int=-1) -> None:
    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION (ARIMA)", highlight=color)
    train_test_split:int = int(len(TS)*params['train_test_split'])
    TS_train = TS[:-params['look_ahead']]
    TS_test= TS[-params['look_ahead']:]
    p = params['arima_p']
    i = params['arima_i']
    q = params['arima_q']
    if verbose:
        print("Initializing ARIMA model with the following parameters:")
        utils.print_colored("\tp", color=color, end=f": {p}\n")
        utils.print_colored("\ti", color=color, end=f": {i}\n")
        utils.print_colored("\tq", color=color, end=f": {q}\n")
        print(f"\tTotal: ", end="")
        utils.print_colored(p+q, color=color)
        print(f"Provided Timeseries has ", end="")
        utils.print_colored(len(TS_train), color=color, end=" ")
        print("realizations.")
        print("Fitting ... ", end="")
    model = tsu.arima_model(p=p,
                            i=i,
                            q=q,
                            train_series=TS_train,
                            verbose=False,
                            color=color,
                           )
    if verbose:
        print("done.")
    forecast_mean, conf_int = tsu.arima_forecast(model=model,
                                              steps=params['look_ahead'],
                                              alpha=params['arima_alpha'],
                                             )
    error = tsu.validate_forecast(historical=TS_train[-plot_limit+params['look_ahead']:],
                                  forecast_mean=forecast_mean,
                                  model_name='ARIMA',
                                  conf_int=conf_int,
                                  ground_truth=TS_test.reshape(-1),
                                  alpha=params['arima_alpha'],
                                  file_path=f"{params['timeseries_folder']}/ARIMA-{p}-{i}-{q}",
                                  verbose=verbose,
                                  color=color,
                                 )
    tsu.diagnostic_check(model,
                         output_folder=params['timeseries_folder'],
                         lag=params['look_ahead'],
                         model_name="ARIMA",
                         verbose=verbose
                        )
    return model


def train_test_AR(params, TS, verbose:bool=True, color:str="blue", plot_limit:int=-1) -> None:
    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION (AR)", highlight=color)
    TS_train = TS[:-params['look_ahead']]
    TS_test= TS[-params['look_ahead']:]
    p = params['arima_p']
    if verbose:
        print("Initializing AR model with the following parameters:")
        utils.print_colored("\tp", color=color, end=f": {p}\n")
        print(f"Provided Timeseries has ", end="")
        utils.print_colored(len(TS_train), color=color, end=" ")
        print("realizations.")
        print("Fitting ... ", end="")
    model = tsu.arima_model(p=p,
                            i=0,
                            q=0,
                            train_series=TS_train,
                            verbose=False,
                            color=color,
                           )
    if verbose:
        print("done.")
    forecast_mean, conf_int = tsu.arima_forecast(model=model,
                                              steps=params['look_ahead'],
                                              alpha=params['arima_alpha'],
                                             )
    error = tsu.validate_forecast(historical=TS_train[-plot_limit+params['look_ahead']:],
                                  model_name='AR',
                                  forecast_mean=forecast_mean,
                                  conf_int=conf_int,
                                  ground_truth=TS_test.reshape(-1),
                                  alpha=params['arima_alpha'],
                                  file_path=f"{params['timeseries_folder']}/AR-{p}",
                                  verbose=verbose,
                                  color=color,
                                 )
    tsu.diagnostic_check(model,
                         output_folder=params['timeseries_folder'],
                         lag=params['look_ahead'],
                         model_name="AR",
                         verbose=verbose
                        )
    return model


def train_test_MA(params, TS, verbose:bool=True, color:str="blue", plot_limit:int=-1) -> None:
    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION (MA)", highlight=color)
    TS_train = TS[:-params['look_ahead']]
    TS_test= TS[-params['look_ahead']:]
    q = params['arima_q']
    if verbose:
        print("Initializing MA model with the following parameters:")
        utils.print_colored("\tq", color=color, end=f": {q}\n")
        print(f"Provided Timeseries has ", end="")
        utils.print_colored(len(TS_train), color=color, end=" ")
        print("realizations.")
        print("Fitting ... ", end="")
    model = tsu.arima_model(p=0,
                            i=0,
                            q=q,
                            train_series=TS_train,
                            verbose=False,
                            color=color,
                           )
    if verbose:
        print("done.")
    forecast_mean, conf_int = tsu.arima_forecast(model=model,
                                              steps=params['look_ahead'],
                                              alpha=params['arima_alpha'],
                                             )
    error = tsu.validate_forecast(historical=TS_train[-plot_limit+params['look_ahead']:],
                                  model_name='MA',
                              forecast_mean=forecast_mean,
                              conf_int=conf_int,
                              ground_truth=TS_test.reshape(-1),
                              alpha=params['arima_alpha'],
                              file_path=f"{params['timeseries_folder']}/MA-{q}",
                              verbose=verbose,
                              color=color,
                             )
    tsu.diagnostic_check(model,
                         output_folder=params['timeseries_folder'],
                         lag=params['look_ahead'],
                         model_name="MA",
                         verbose=verbose
                        )
    return model


def train_test_ARCH(params, TS, verbose:bool=True, color:str="blue", plot_limit:int=-1) -> None:
    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION (ARCH)", highlight=color)
    TS_train = TS[:-params['look_ahead']]
    TS_test= TS[-params['look_ahead']:]
    p = params['arch_p']
    lag = params['arch_lag']
    if verbose:
        print("Initializing ARCH model with the following parameters:")
        utils.print_colored("\tp", color=color, end=f": {p}\n")
        print(f"Provided Timeseries has ", end="")
        utils.print_colored(len(TS_train), color=color, end=" ")
        print("realizations.")
        print("Fitting ... ", end="")
    model = tsu.volatility_model(series=TS_train,
                                 p=p, q=0,
                                 model_type='ARCH',
                                 lags=lag,
                                 mean=params['volatility_mean'],
                                 dist=params['volatility_dist'],
                                )
    if verbose:
        print("done.")
    forecast_mean, forecast_variance = tsu.forecast_volatility(model, steps=params['look_ahead'])
    conf_int = np.concatenate([(forecast_mean+forecast_variance).reshape(-1,1), (forecast_mean-forecast_variance).reshape(-1,1)], axis=1)
    # TODO: this
    error = tsu.validate_forecast(historical=TS_train[-plot_limit+params['look_ahead']:],
                                  model_name='ARCH',
                                  forecast_mean=forecast_mean,
                                  conf_int=conf_int,
                                  ground_truth=TS_test.reshape(-1),
                                  file_path=f"{params['timeseries_folder']}/ARCH-{p}-{lag}",
                                  verbose=verbose,
                                  color=color,
                                 )
    tsu.diagnostic_check(model,
                         output_folder=params['timeseries_folder'],
                         lag=params['look_ahead'],
                         model_name="ARCH",
                         verbose=verbose
                        )
    return model


def train_test_GARCH(params, TS, verbose:bool=True, color:str="blue", plot_limit:int=-1) -> None:
    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION (GARCH)", highlight=color)
    TS_train = TS[:-params['look_ahead']]
    TS_test= TS[-params['look_ahead']:]
    p = params['arch_p']
    q = params['garch_q']
    lag = params['arch_lag']
    if verbose:
        print("Initializing GARCH model with the following parameters:")
        utils.print_colored("\tp", color=color, end=f": {p}\n")
        utils.print_colored("\tq", color=color, end=f": {q}\n")
        print(f"Provided Timeseries has ", end="")
        utils.print_colored(len(TS_train), color=color, end=" ")
        print("realizations.")
        print("Fitting ... ", end="")
    model = tsu.volatility_model(series=TS_train,
                                 p=p, q=q,
                                 model_type='GARCH',
                                 lags=lag,
                                 mean=params['volatility_mean'],
                                 dist=params['volatility_dist'],
                                )
    if verbose:
        print("done.")
    forecast_mean, forecast_variance = tsu.forecast_volatility(model, steps=params['look_ahead'])
    conf_int = np.concatenate([(forecast_mean+forecast_variance).reshape(-1,1), (forecast_mean-forecast_variance).reshape(-1,1)], axis=1)
    # TODO: this
    error = tsu.validate_forecast(historical=TS_train[-plot_limit+params['look_ahead']:],
                                  model_name='ARCH',
                                  forecast_mean=forecast_mean,
                                  conf_int=conf_int,
                                  ground_truth=TS_test.reshape(-1),
                                  file_path=f"{params['timeseries_folder']}/GARCH-{p}-{lag}-{q}",
                                  verbose=verbose,
                                  color=color,
                                 )
    tsu.diagnostic_check(model,
                         output_folder=params['timeseries_folder'],
                         lag=params['look_ahead'],
                         model_name="GARCH",
                         verbose=verbose
                        )
    return model


    

