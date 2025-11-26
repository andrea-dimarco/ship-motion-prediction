import utils.utils as utils

from multivariate.multivariate_deep_learning import *
from multivariate.multivariate_timeseries import *


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    # PARAMETERS
    color:str = "magenta"
    plot_limit:int = 100

    params:dict = utils.load_json("/data/params.json")
        
    deep_learning_n_to_1(params, plot_limit=plot_limit, color=color)
