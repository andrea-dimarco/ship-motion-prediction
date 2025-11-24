
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from typing import Literal, Any, Callable

import utils.nn_utils as nnu
import utils.utils as utils
import utils.plot_utils as plu

from multivariate.multivariate_deep_learning import *
from multivariate.multivariate_timeseries import *


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    # PARAMETERS
    color:str = "magenta"
    plot_limit:int = 100

    params:dict = utils.load_json("/data/params.json")
    
    multivariate_timeseries_analysis(params, plot_limit=plot_limit, color=color)
    
    deep_learning_model(params, plot_limit=plot_limit, color=color)
