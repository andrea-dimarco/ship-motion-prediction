
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Literal, Any, Callable

import utils.utils as utils
import utils.plot_utils as plu
import utils.timeseries_utils as tsu
import data_handling as dh



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
    train_test_split:int = int(len(TS)*params['train_test_split'])
    model = tsu.arima_model(p=params['arima_p'],
                            i=params['arima_i'],
                            q=params['arima_q'],
                            train_series=TS[:train_test_split],
                            verbose=verbose,
                            color=color,
                           )
    # TODO: bug fix
    forecast = tsu.forecast_expected_value(model=model,
                                           timeseries=TS[train_test_split:],
                                           n_periods=1,
                                          )
    tsu.plot_forecast(time_series=TS[train_test_split:],
                      forecast=forecast,
                      output_folder=params['timeseries_folder'],
                      verbose=verbose,
                      model_name=f"ARIMA-{params['arima_p']}-{params['arima_i']}-{params['arima_q']}"
                     )
    

