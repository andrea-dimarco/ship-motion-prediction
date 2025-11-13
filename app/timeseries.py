
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Literal, Any, Callable

import utils.utils as utils
import utils.plot_utils as plu
import utils.timeseries_utils as tsu
import data_handling as dh


def forecast_future(model, steps=10, alpha=0.05):
    """
    Forecast future values from a fitted ARIMA model that was fit on a numpy array.
    Args:
        fitted_model: result from fit_arima_numpy(...)
        y: original 1-D numpy array (used for shape/length reference).
        steps: number of future periods to forecast.
        alpha: significance level for confidence intervals.
    Returns:
        forecast_mean: 1-D numpy array of length = steps.
        ci: 2-D numpy array of shape (steps, 2) with lower and upper bounds. (confident interval)
    """
    # Depending on statsmodels version, you may use .get_forecast or .forecast
    try:
        fc_obj = model.get_forecast(steps=steps)
        forecast_mean = fc_obj.predicted_mean
        ci = fc_obj.conf_int(alpha=alpha)
        # If original fit used numpy array, ci may be numpy array too.
    except AttributeError:
        # fallback for older statsmodels
        forecast_mean, stderr, ci = model.forecast(steps=steps, alpha=alpha)
        # Here ci is a numpy array of shape (steps,2)
    # Ensure numpy array output
    forecast_mean = np.asarray(forecast_mean).flatten()
    ci = np.asarray(ci)
    if ci.ndim == 1:
        ci = ci.reshape(-1, 2)
    return forecast_mean, ci



def validate_forecast(historical:np.ndarray,
                      forecast_mean:np.ndarray,
                      conf_int:np.ndarray,
                      file_path:str|None=None,
                      ground_truth:np.ndarray|None=None,
                      alpha:float=0.05,
                      verbose:bool=True,
                      color:str="blue",
                     ) -> float:
    """
    Plot original numpy-series + forecast + confidence interval.
    """
    # COMPUTE ERROR
    import torch
    from torch import nn
    error = float(nn.L1Loss()(torch.from_numpy(forecast_mean), torch.from_numpy(ground_truth)))
    if verbose:
        print("Forecasting", end=" ")
        utils.print_colored(len(forecast_mean), color=color, end=" ")
        print("steps ahead gave an error of", end=" ")
        utils.print_colored(error, color=color)
    if file_path is not None:
        plt.figure(figsize=(10,6))
        # plot original
        plt.plot(np.arange(len(historical)), historical, label='Historical')
        # plot forecast
        start = len(historical)
        steps = np.arange(start,start+len(forecast_mean))
        if ground_truth is not None:
            assert len(ground_truth) == len(forecast_mean)
            plt.plot(steps, ground_truth, label='Ground Truth', color='black', linewidth=2)
            plt.fill_between(steps, ground_truth, forecast_mean,
                            where=None,       # or a boolean array if you only want some segments
                            interpolate=True, # helps when lines cross
                            color='red',
                            alpha=0.3,
                            label="Error",
                            )
        plt.plot(steps, forecast_mean, label='Forecast', color='red', linestyle="--")
        # fill between CI
        lower = conf_int[:,0]
        upper = conf_int[:,1]
        plt.fill_between(np.arange(start, start + len(forecast_mean)), lower, upper,
                        color='cyan',
                        alpha=0.3,
                        label=f'{100*(1-alpha):.1f}% Confidence Interval'
                       )
        plt.legend()
        plt.grid()
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.title(f"Arima prediction of {len(forecast_mean)} steps ahead (err:{round(error,5)})")
        plt.savefig(file_path)
    return error



def stationarity_analysis(DF, plot_folder:str|None=None, max_diff:int|None=None, verbose:bool=True, color:str="blue", plot_limit:int=-1) -> np.ndarray:
    '''
    Checks stationary features and cointegration opportunities, then differentiate the data until all features are stationary and returns it
    '''
    stationarity_info = tsu.non_stationary_features_list(multivariate_timeseries=DF.to_numpy(),
                                                         features=list(DF.columns),
                                                         detailed_info=True,
                                                         verbose=verbose
                                                        )
    stationary, diff = tsu.make_multivariate_diff_stationary(multivariate_timeseries=DF.to_numpy(),
                                                             features=list(DF.columns),
                                                             verbose=verbose,
                                                             max_diff=max_diff,
                                                            )
    cointegration = tsu.check_multivariate_cointegration(multivariate_timeseries=stationary,#DF.to_numpy(),
                                                         features=list(DF.columns),
                                                         verbose=verbose,
                                                         color=color,
                                                        )
    if plot_folder is not None:
        fig = plt.figure(figsize=(8, 7))
        UP = fig.add_subplot(2, 1, 1)
        DOWN = fig.add_subplot(2, 1, 2)
        # Plot timeseries
        UP.plot(DF[:plot_limit].to_numpy(), label=list(DF.columns))
        DOWN.plot(stationary[:plot_limit], label=list(DF.columns))
        # Stylize the plot
        UP.grid()
        DOWN.grid()
        UP.set_title("BEFORE Stationary Transformations")
        DOWN.set_title("AFTER Stationary Transformation")
        UP.legend()
        DOWN.legend()
        DOWN.set_xlabel("Timestep")
        plt.savefig(f"{plot_folder}stationarity_check.png")
        plt.clf()
    return stationary



def timeseries_analysis(params:dict, plot_limit:int=-1, color:str="blue") -> None:
    '''
    Performs the timeseries analysis and model fit on the data available
    '''
    verbose:bool = params['verbose']

    features = params['input_features']
    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION (TIMESERIES)", highlight=color)
        print("Features:")
        utils.print_two_column(params['input_features'], color=color)
    
    # RAW DATA
    DF = dh.load_dataset(f"{params['dataset_folder']}/{params['dataset']}",
                         features=set(features),
                         verbose=verbose,
                         normalize=True,
                        )
    # STATIONARITY
    if params['enforce_stationarity']:
        TS = stationarity_analysis(DF=DF,
                                   max_diff=None,
                                   verbose=verbose,
                                   color=color,
                                   plot_folder=params['timeseries_folder'],
                                   plot_limit=plot_limit,
                                  )
    else:
        TS = DF.to_numpy()
    
    # ARIMA Model
    assert len(params['input_features']) == 1
    train_test_split:int = int(len(TS)*params['train_test_split'])
    TS_train = TS[:train_test_split]
    TS_test= TS[train_test_split:]
    p = params['arima_p']
    i = params['arima_i']
    q = params['arima_q']
    if verbose:
        print("Initializing ARIMA model with the following parameters:")
        utils.print_colored("\tp", color=color, end=f": {p}\n")
        utils.print_colored("\ti", color=color, end=f": {i}\n")
        utils.print_colored("\tq", color=color, end=f": {q}\n")
        print(f"\tTotal: {p+q}")
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
    forecast_mean, conf_int = forecast_future(model=model,
                                              steps=params['look_ahead'],
                                              alpha=params['arima_alpha'],
                                             )
    error = validate_forecast(historical=TS_train[-plot_limit+params['look_ahead']:],
                              forecast_mean=forecast_mean,
                              conf_int=conf_int,
                              ground_truth=TS_test.reshape(-1)[:params['look_ahead']],
                              alpha=params['arima_alpha'],
                              file_path=f"{params['timeseries_folder']}/ARIMA-{p}-{i}-{q}",
                              verbose=verbose,
                              color=color,
                             )
    


def arima_parameter_estimation() -> tuple[int, int, int]:
    pass
    # TODO: this
    

