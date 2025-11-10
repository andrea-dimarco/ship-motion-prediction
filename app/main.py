
import os
from typing import Literal, Any, Callable

import utils.utils as utils




if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    # PARAMETERS
    color:str = "yellow"

    params:dict = utils.load_json("/data/params.json")
    
    