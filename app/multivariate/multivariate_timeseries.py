
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Literal, Any, Callable

import utils.utils as utils
import utils.plot_utils as plu
import utils.timeseries_utils as tsu
import data_handling as dh



def multivariate_timeseries_analysis(params:dict, plot_limit:int=-1, color:str="blue") -> None:
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
    
    # VAR Model
    DF = DF[sorted(list(DF.columns))]
    # from utils.plot_utils import corr_heatmap
    # corr_heatmap(correlation=DF.corr(),
    #              show_pic=False,
    #              save_pic=True,
    #              pic_name=f"{params['dataset']}-correlation",
    #              pic_folder=f"{params['img_folder']}"
    #             )
    train_test_VAR(params, DF, verbose, color, plot_limit=plot_limit)

    
# # # # # # # # # #
# MODEL FUNCTIONS #
# # # # # # # # # #



def train_test_VAR(params, TS:pd.DataFrame, verbose:bool=True, color:str="blue", plot_limit:int=-1) -> None:
    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION (VAR)", highlight=color)
    train_test_split:int = int(len(TS)*params['train_test_split'])
    TS_train = TS[:-params['look_ahead']]
    TS_test= TS[-params['look_ahead']:]
    p = params['varmax_p']
    n_features = len(list(TS.columns))
    if verbose:
        print("Initializing VAR model with the following parameters:")
        utils.print_colored("\tp", color=color, end=f": {p}\n")
        utils.print_colored("\ttimeseries", color=color, end=f": {n_features}\n")
        print(f"\tTotal: ", end="")
        utils.print_colored(p*(n_features**2), color=color)
        print(f"Provided Timeseries has ", end="")
        utils.print_colored(len(TS_train), color=color, end=" ")
        print("realizations.")
        print("Fitting ... ", end="")
    model = tsu.var_model(p=p,
                          train_series=TS_train.to_numpy(),
                          verbose=False,
                         )
    if verbose:
        print("done.")
    forecast = tsu.var_forecast(model=model,
                                input_samples=TS_train[-p:],
                                steps=params['look_ahead'],
                               )
    # VALIDATE
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    for col in TS_test.columns:
        mae = mean_absolute_error(TS_test[col], forecast[col])
        rmse = np.sqrt(mean_squared_error(TS_test[col], forecast[col]))
        if verbose:
            utils.print_colored(f"\t{col}", color=color, end=" --> ")
            print(f"MAE: {mae:.4f} -- RMSE: {rmse:.4f}")
    from utils.plot_utils import confront_multivariate_plots
    if verbose:
        print("Plotting VAR forecasting ... ", end="")
    confront_multivariate_plots(main_series=TS_test.to_numpy(), main_label='Actual',
                                other_series=forecast.to_numpy(), other_label='Predicted',
                                title=f"VAR({p}) {params['look_ahead']} steps forecasting",
                                plot_img=f"{params['timeseries_folder']}/VAR-{p}-{n_features}",
                                labels=list(TS_test.columns),
                               )
    if verbose:
        print("done.")
    return model





    

