
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from typing import Literal, Any, Callable

import utils.nn_utils as nnu
import utils.utils as utils



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
    


def get_data(task:Literal['REGR','CLAS'],
             seq_len:int,
             dataset_file:str,
             in_features:set[str],
             out_features:set[str]|None=None,
             n_labels:int|None=None,
             verbose:bool=True
            ) -> tuple[torch.Tensor,torch.Tensor]:
    import data_handling as dh
    from numpy import pi
    if out_features is None:
        out_features = in_features.copy()
    DF = dh.load_dataset(file_path=dataset_file,
                         features=in_features.union(out_features),
                         verbose=verbose,
                         normalize=True,
                        )
    if task == 'REGR':
        X, Y = dh.build_sequences(df=DF,
                                  X_features=in_features,
                                  Y_features=out_features,
                                  seq_len=seq_len,
                                  verbose=verbose
                                 )
    elif task == 'CLAS':
        assert len(out_features) == 1, "Multi-label samples are not supported (yet)"
        label_feature = list(out_features)[0]
        DF = dh.add_label_to_timeseries(DF, n_labels=n_labels, label_feature=label_feature)
        X, Y = dh.build_sequences_labels(df=DF,
                                         seq_len=seq_len,
                                         labels_columns=[f'label_{label_feature}'],
                                         verbose=verbose
                                        )
        Y = dh.tensor_seq_to_one_hot(Y, num_classes=n_labels)
    else:
        raise ValueError(f"Unsupported task ({task})")
    # only get last element of sequence for training
    Y = Y[:, -1, :]  # shape (n_sequences, sample_size)
    return X, Y
    


def train_test_split(X:torch.Tensor, y:torch.Tensor, split:float=0.75) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Splits the two tensors `X` and `y` in two at the same cutting point defined by `split`
    '''
    assert split > 0.0 and split < 1.0
    split_index = int(X.shape[0] * split)
    
    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test



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
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='black', linewidth=2)
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_img)



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
            y_pred, hidden = model(current_seq, hidden)
        except TypeError:
            # GRU
            y_pred = model(current_seq)
        # y_pred shape: (batch, out_dim)
        predictions.append(y_pred.unsqueeze(1))  # (batch,1,out_dim)
        # 2) prepare the next input sequence by appending y_pred and dropping oldest.
        # If your input size in_dim == out_dim, you can directly use it; if not, you may need a mapping
        next_input = y_pred.unsqueeze(1)  # (batch,1,out_dim)
        # drop first time step, shift sequence left, append next_input
        current_seq = torch.cat( (current_seq[:,1:,:], next_input), dim=1 )
    return torch.cat(predictions, dim=1)  # (batch, n_steps, out_dim)



def multi_step_forecast_plot(model:(...),
                             X:torch.Tensor,
                             y:torch.Tensor,
                             n_steps:int=5,
                             img_path:str|None=None,
                            ) -> None:
    '''
    Plots the multi-step forecast results

    **Arguments**:
    - `model` : the trained MyLSTM model
    - `X` : tensor of shape (`batch`, `seq_len`, `in_dim`)
    - `y` : tensor of shape (`batch`, `seq_len`, `out_dim`)
    - `n_steps` : integer number of future steps to predict
    '''
    # pick batch instance and feature dimension
    feat = 0
    y_pred = multi_step_forecast(model, X, n_steps)
    # convert tensors to numpy if needed
    y_true_np = y[:,feat].cpu().detach().numpy()   # shape (n_steps,)
    y_pred_np = y_pred[:,feat].cpu().detach().numpy()   # shape (n_steps,)

    # create x‐axis for steps: you can choose e.g. from 1→n_steps
    steps = np.arange(1, len(y_true_np)+1)

    plt.figure(figsize=(10,6))
    plt.plot(steps, y_true_np, label='Actual', marker='o')
    plt.plot(steps, y_pred_np, label='Predicted', marker='x', linestyle='--')
    plt.xlabel('Future Step')
    plt.ylabel(f'Feature {feat} value')
    plt.title('Actual vs Predicted - Multi-step forecast')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if img_path is not None:
        plt.savefig(img_path)



if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    # PARAMETERS
    color:str = "magenta"
    plot_limit:int = 200
    look_ahead:int = 10

    params:dict = utils.load_json("/data/params.json")
    verbose:bool = params["verbose"]
    case_study:str = f"{params['model']}-{params['task']}"

    if verbose:
        utils.print_colored(f"SHIP-MOTION PREDICTION ({case_study})", highlight=color)
        print("Input Features:")
        utils.print_two_column(params['input_features'], color=color)
        print("Output Features")
        utils.print_two_column(params['output_features'], color=color)
    
    X, Y = get_data(task=params['task'],
                    seq_len=params['seq_len'],
                    dataset_file=f"{params['dataset_folder']}/{params['dataset']}",
                    in_features=set(params['input_features']),
                    out_features=set(params['output_features']),
                    n_labels=params['n_classes'],
                    verbose=verbose
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
    
    plot_predictions(model=model,
                     X_test=X_test[:plot_limit],
                     y_test=y_test[:plot_limit],
                     plot_img=f"{result_folder}/{case_study}-prediction.png",
                    )
    
    multi_step_forecast_plot(model=model,
                             X=X_test[:plot_limit],
                             y=y_test[:plot_limit],
                             n_steps=look_ahead,
                             img_path=f"{result_folder}/{case_study}-look_ahead.png",
                            )