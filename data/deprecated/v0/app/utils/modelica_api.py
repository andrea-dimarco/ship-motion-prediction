import pandas as pd

import os
from typing import Any, Literal

import utils.utils as utils


class ModelicaAPI:

    def __init__(self, *,
                 model_folder:str,
                 simulation_horizon:int=100000,
                 solver:Literal["dassl","rungekutta"]="dassl",
                 quiet:bool=False,
                 model_name:str="System"
                ):
        self.__model_folder:str = model_folder
        self.__simulation_horizon:int = simulation_horizon
        self.__solver:str = solver
        self.__quiet:bool = quiet
        self.__model_name:str = model_name

    # GETTER & SETTER METHODS
    def name(self) -> str:
        return self.__model_name
    
    def quiet(self, new:bool|None=None) -> bool:
        if new is not None:
            self.__quiet = new
        return self.__quiet

    def solver(self, new:str|None=None) -> str:
        if new is not None:
            self.__solver = new
        return self.__solver

    def folder(self, new:str|None=None) -> str:
        if new is not None:
            self.__model_folder = new
        return self.__model_folder
    
    def time_limit(self, new:int|None=None) -> str:
        if new is not None:
            self.__simulation_horizon = new
        return self.__simulation_horizon


    # SIMULATION METHODS
    def compile(self) -> None:
        """
        Runs the commands necessary to compile the model.
        """
        if not self.quiet():
            print("Compiling model ...", end="")
        os.system(f"cd {self.folder()}; ./run.sh >> log")
        os.system(f"cd {self.folder()}; rm {self.name()}_res.* >> log")
        files, _ = utils.scan_dir(self.folder())
        if self.name() not in files:
            raise SystemError(f"Model either not compiled or compiled with the wrong output format, please read the log file in {self.folder()}") 
        if not self.quiet():
            print("done.")


    def delete_binaries(self) -> None:
        """
        Deletes the compiled Modelica files
        """
        if not self.quiet():
            print("Deleting compiled files ...", end="")
        cmd = f"cd {self.folder()}; ./clean.sh >> log"
        result = os.popen(cmd).read()
        if not self.quiet():
            print("done.")


    def run(self,
            output_variables:set[str],
            parameters:dict=None,
           ) -> dict:
        """
        Run the model and return the model output as a dictionary

        **Arguments:**
        - `output_variables`: *set* of output variables to return from the model execution
        - `parameters`: dictionary with assignment for the model parameters
        """
        # SELECT OUTPUT VARIABLES
        output_str = ""
        for var in sorted(list(output_variables))[:-1]:
            output_str += f"{var},"
        output_str += sorted(list(output_variables))[-1]
        # PARAMETER UPDATE
        if parameters is not None:
            param_str = ""
            for param_name in parameters.keys():
                param_value = parameters[param_name]
                param_str += f'params.{param_name}={param_value},'
            param_str += f"params.time_horizon={self.time_limit()}"

            # RUN MODEL
            cmd = f"cd {self.folder()}; ./{self.name()} -s={self.solver()} -noemit -override={param_str} -output={output_str}"
            result = os.popen(cmd).read()
        else:
            # RUN MODEL
            cmd = f"cd {self.folder()}; ./{self.name()} -s={self.solver()} -noemit -output={output_str}"
            result = os.popen(cmd).read()
        # GET RESULTS
        result_dict = dict()
        for line in result.split('\n'):
            if line.startswith('time'):
                for var_res in line.split(','):
                    var, value = var_res.split('=')
                    result_dict[var] = float(value)
        
        return result_dict






if __name__ == '__main__':
    utils.print_colored("MODELICA API Script", highlight="magenta")

    verbose:bool = True
    parameters:dict = utils.load_json("/data/marl_params.json")
    case_study_params:dict = parameters["case_study_params"][parameters["case_study"]]

    time_horizon:int = 14400

    # CUSTOM API
    utils.print_colored("Custom API", highlight="blue")
    model = ModelicaAPI(model_folder="/app/modelica_env/", simulation_horizon=time_horizon, quiet=not verbose)
    model.compile()
    res = model.run(output_variables={'evader_0.x', 'evader_0.y', 'evader_0.z'})
    model.delete_binaries()
    utils.print_dict(res, key_color="blue")
    if int(res['time']) != time_horizon:
        utils.print_colored(f"Simulation ended abruptly at time {int(res['time'])} instead of time {time_horizon}", color="red")

    print("Test completed.")