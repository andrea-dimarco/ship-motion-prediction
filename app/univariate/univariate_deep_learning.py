
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from typing import Literal, Any, Callable

import utils.nn_utils as nnu
import utils.utils as utils
import utils.plot_utils as plu
from data_handling import get_data, train_test_split



def get_predictor(type:Literal['LSTM','GRU'],
                  task:Literal['REGR','CLAS'],
                  input_dim:int,
                  output_dim:int,
                  num_layers:int,
                  hidden_dim:int,
                 ) -> Any:
    assert type in {'LSTM','GRU'}
    assert task in {'REGR','CLAS'}
    if type == 'LSTM':
        if task == 'REGR':
            from networks.LSTM_regressor import LSTMRegressor
            return LSTMRegressor(in_dim=input_dim,
                                 out_dim=output_dim,
                                 num_layers=num_layers,
                                 hidden_dim=hidden_dim,
                                )
        elif task == 'CLAS':
            from networks.LSTM_classifier import LSTMClassifier
            return LSTMClassifier(in_dim=input_dim,
                                  out_dim=output_dim,
                                  num_layers=num_layers,
                                  hidden_dim=hidden_dim,
                                 )
        else:
            raise ValueError(f"Unsupported task ({task})")
        
    elif type == 'GRU':
        if task == 'REGR':
            from networks.GRU_regressor import GRURegressor
            return GRURegressor(in_dim=input_dim,
                                out_dim=output_dim,
                                num_layers=num_layers,
                                hidden_dim=hidden_dim,
                               )
        elif task == 'CLAS':
            from networks.GRU_classifier import GRUClassifier
            return GRUClassifier(in_dim=input_dim,
                                 out_dim=output_dim,
                                 num_layers=num_layers,
                                 hidden_dim=hidden_dim,
                                )
        else:
            raise ValueError(f"Unsupported task ({task})")
        
    else:
        raise ValueError(f"Unsupported predictor ({type})")
    


def plot_predictions(model:(...), X_test:torch.Tensor, y_test:torch.Tensor, plot_img:str, title:str="Predictions vs Actual"):
    '''
    Plots the model's predictions against the actual `y_test` values.

    Parameters:
    - `model` : Trained callable predictor
    - `X_test` : Test input data (3D array: [n_samples, seq_len, sample_dim])
    - `y_test` : True target values corresponding to `X_test`
    - `plot_img` : where to save the picture
    - `title` : Title of the plot.
    '''

    # Generate predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        # Flatten arrays if necessary (useful for single-output models)
        y_pred = y_pred.flatten()
        y_true = y_test.flatten()
    # Create the plot
    from utils.plot_utils import confront_univariate_plots
    confront_univariate_plots(main_series=y_true.numpy(),
                              other_series=y_pred.numpy(),
                              main_label='Actual',
                              other_label='Predicted',
                              title='Prediction vs Actual',
                              plot_img=plot_img
                             )



def get_loss(task:str) -> (...):
    '''
    Returns the los function to use during training
    '''
    if task == 'CLAS':
        return nn.CrossEntropyLoss() 
    elif task == 'REGR':
        return nn.L1Loss() # nn.MSELoss()
    else:
        raise ValueError(f"Unsupported task ({task})")



def multi_step_forecast(model:(...), init_seq:torch.Tensor, n_steps:int=1) -> torch.Tensor:
    """
    **Arguments**:
    - `model` : the trained MyLSTM model
    - `init_seq` : tensor of shape (`batch`, `seq_len`, `in_dim`) — the last observed input
    - `n_steps` : integer number of future steps to predict
    
    **Returns**:
    - tensor of shape (batch, n_steps, out_dim)
    """
    model.eval()
    batch, seq_len, in_dim = init_seq.shape
    predictions = []
    current_seq = init_seq.clone().detach()
    hidden = None
    for step in range(n_steps):
        # 1) predict next step
        try:
            # LSTM
            y_pred, hidden = model(current_seq, None) # TODO: hidden or None? since the sequence is complete
        except TypeError:
            # GRU
            y_pred = model(current_seq)
        # y_pred shape: (batch, out_dim)
        predictions.append(y_pred.unsqueeze(1))  # (batch,1,out_dim)
        # 2) prepare the next input sequence by appending y_pred and dropping oldest.
        # If your input size in_dim == out_dim, you can directly use it; if not, you may need a mapping
        next_input = y_pred.unsqueeze(1)  # (batch,1,out_dim)
        # drop first time step, shift sequence UP, append next_input
        current_seq = torch.cat( (current_seq[:,1:,:], next_input), dim=1 )
    return torch.cat(predictions, dim=1)  # (batch, n_steps, out_dim)



def multi_step_forecast_validation(model:(...),
                                   X:torch.Tensor,
                                   y:torch.Tensor,
                                   n_steps:int=5,
                                   img_path:str|None=None,
                                   verbose:bool=True,
                                   color:str="blue",
                                  ) -> float:
    '''
    Plots the multi-step forecast results

    **Arguments**:
    - `model` : the trained MyLSTM model
    - `X` : tensor of shape (`batch`, `seq_len`, `in_dim`)
    - `y` : tensor of shape (`batch`, `seq_len`, `out_dim`)
    - `n_steps` : integer number of future steps to predict
    '''
    # pick batch instance and feature dimension
    feat = 0 # TODO: this must be removed as there is only one feature
    y_pred = multi_step_forecast(model, X, n_steps) # (n_seq, n_steps, out_dim)

    # format data for plot
    y_true_np = y[n_steps-1:,:].cpu().detach().numpy().reshape(-1) # take the actual value n_steps ahead (-1 because y is already 1 step ahead)
    y_pred_np = y_pred[:len(y_true_np),-1,:].cpu().detach().numpy().reshape(-1) # (n_seq, out_dim) # NOTE: only take the n_step ahead prediction
    
    # COMPUTE ERROR
    error = float(nn.L1Loss()(torch.from_numpy(y_pred_np), torch.from_numpy(y_true_np)))
    if verbose:
        print("Forecasting", end=" ")
        utils.print_colored(n_steps, color=color, end=" ")
        print("steps ahead gave an error of", end=" ")
        utils.print_colored(error, color=color)

    # create x‐axis for steps: you can choose e.g. from 1→n_steps
    steps = np.arange(1, len(y_true_np)+1)
    if img_path is not None:
        plt.figure(figsize=(10,6))
        plt.plot(steps, y_true_np, label='Actual', marker='o')
        plt.plot(steps, y_pred_np, label='Predicted', marker='x', linestyle='--')
        plt.fill_between(steps, y_true_np, y_pred_np,
                        where=None,       # or a boolean array if you only want some segments
                        interpolate=True, # helps when lines cross
                        color='red',
                        alpha=0.3,
                        label="Error",
                        )
        plt.xlabel('Future Step')
        plt.ylabel(f'Feature {feat} value')
        plt.title(f'Actual vs Predicted - {n_steps}-step forecast (err:{round(error,5)})')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(img_path)
    return error



def deep_learning_model(params:dict, plot_limit:int=-1, color:str="blue") -> None:
    verbose:bool = params["verbose"]
    case_study:str = f"{params['model']}-{params['task']}"

    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION ({case_study})", highlight=color)
        print("Input Features:")
        utils.print_two_column(params['input_features'], color=color)
        print("Output Features:")
        utils.print_two_column(params['output_features'], color=color)
    
    X, Y = get_data(task=params['task'],
                    seq_len=params['seq_len'],
                    dataset_file=f"{params['dataset_folder']}/{params['dataset']}",
                    in_features=set(params['input_features']),
                    out_features=set(params['output_features']),
                    n_labels=params['n_classes'],
                    verbose=verbose,
                    reduce_frequency=params['reduce_frequency'],
                   )
    X_train, y_train, X_test, y_test = train_test_split(X=X, y=Y, split=params['train_test_split'])

    if verbose:
        print(f"Retrieving model for ", end="")
        utils.print_colored(case_study, color=color, end=" ... ")

    model = get_predictor(input_dim=len(params['input_features']), # TODO: this will change with the actual data
                          output_dim=len(params['output_features']) if params['task'] == 'REGR' else params['n_classes'],
                          type=params['model'],
                          task=params['task'],
                          hidden_dim=params['lstm_hidden_dim'] if params['model'] == 'LSTM' else params['gru_hidden_dim'],
                          num_layers=params['lstm_num_layers'] if params['model'] == 'LSTM' else params['gru_num_layers'],
                         )
    if verbose:
        print("done.")

    result_folder:str = params['lstm_folder'] if params['model'] == 'LSTM' else params['gru_folder']
    
    model = nnu.train_model(model=model,
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_test,
                            y_val=y_test,
                            n_epochs=params['n_epochs'],
                            batch_size=params['batch_size'],
                            loss_plot_folder=result_folder,
                            model_name=case_study,
                            loss_fn=get_loss(task=params['task']),
                            val_frequency=params['validation_frequency'],
                            save_folder=result_folder,
                            verbose=verbose,
                            adam_lr=params['adam_lr'],
                            adam_b1=params['adam_b1'],
                            adam_b2=params['adam_b2'],
                            decay_start=params['decay_start'],
                            decay_end=params['decay_end'],
                            color=color,
                           ) 
    
    if params['task'] == 'REGR':
        plot_predictions(model=model,
                        X_test=X_test[:plot_limit],
                        y_test=y_test[:plot_limit],
                        plot_img=f"{result_folder}/{case_study}-prediction.png",
                        )
        multi_step_error = multi_step_forecast_validation(model=model,
                                                        X=X_test[:plot_limit],
                                                        y=y_test[:plot_limit],
                                                        n_steps=params['look_ahead'],
                                                        img_path=f"{result_folder}/{case_study}-look_ahead.png",
                                                        color=color,
                                                        )
    else:
        raise ValueError(f"Unsupported task ({params['task']})")


