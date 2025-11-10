
import os
from typing import Literal, Any, Callable
from suv.trajectory import Traiettoria_Scitech

import numpy as np
import nevergrad as ng

from optimization.bbo_nevergrad import run_bbo, get_optimizer
import utils.utils as utils
from suv.quad import Quad



def evaluate_simulation(result:Any,
                        trajectory=Traiettoria_Scitech(),
                        t_f:float=10.0,
                        dt:float=0.01,
                        alpha:float=0.5,
                       ) -> float:
    '''
    Computes the KPI
    '''
    pos, vel, quat, omega = result
    # GET TRAJECTORY STEPS
    num_steps = int(t_f / dt) + 1
    traj_pos = np.empty((3, num_steps))
    traj_vel = np.empty((3, num_steps))
    for i in range(num_steps):
        tmp = trajectory.TrajectoryEval(i*dt)
        traj_pos[:,i] = tmp[:3]
        traj_vel[:,i] = tmp[3:]
    # AVERAGE ABSOLUTE ERROR in following the trajectory   
    avg_pos_error:float = abs(np.average(np.average(a=pos-traj_pos, axis=1), axis=0))
    avg_vel_error:float = abs(np.average(np.average(a=vel-traj_vel, axis=1), axis=0))
    # KPI
    return (alpha)*avg_pos_error + (1-alpha)*avg_vel_error



def obj_function(Kpos:float, Kvel:float, Kp:float, Kd:float,
                 saturation:float=1000000,
                 use_constraints:bool=False,
                 t_f:float=5.0,
                 dt:float=0.01,
                ) -> float|tuple[float,list[float]]:
    """
    This function takes the parameters from Nevergrad, formats them and computes 
    the loss function **to minimize**.
    
    **Returns**: `kpi`, `constraints`
    """
    # FORMAT PARAMETERS
    parameters:dict = { "Kpos":Kpos, "Kvel":Kvel, "Kp":Kp, "Kd":Kd }
    data = process_params(parameters)
    # SIMULATE
    trajectory=Traiettoria_Scitech()
    quad = Quad(data, trajectory=trajectory)
    result = quad.run(data=data,
                      t_f=t_f,
                      dt=dt,
                     )
    # COMPUTE KPI
    kpi = evaluate_simulation(result=result,
                              trajectory=trajectory,
                              t_f=t_f,
                              dt=dt,
                             )
    # END
    kpi = utils.clip_value(kpi, minimum=-saturation, maximum=saturation) if not np.isnan(kpi) else saturation
    if use_constraints:
        # CONSTRAINTS
        # TODO: this!
        c_1 = constraint_template(None)
        constraint_violations = [c_1]
        return kpi, constraint_violations
    else: 
        return kpi



def process_params(parameters:dict) -> dict:
    '''
    **Arguments**:
    - `parameters` : dictionary of parameters suggetions from Nevergrad

    **Returns**:
    - `data` : dictionary of parameters formatted as the simulator requires
    '''
    data = utils.load_yaml("/data/quad.yaml")
    data.update(parameters)
    data['Kpos'] *= -1
    data['Kvel'] *= -1
    # data['Kp']
    # data['Kd']
    return data



def constraint_template(model_results:Any) -> float:
    '''
    How to interpret this function

    - \>= 0 if ok

    - < 0 if violated
    '''
    return 0.0



def nvg_instrumentation(verbose:bool=False, MAX:float=100) -> ng.p.Instrumentation:
    """
    Returns the parameters formatted as Nevergrad requires.
    """ 
    if verbose:
        print("Loading instrumentation...", end="")
    instrumentation = ng.p.Instrumentation(  
            Kpos = ng.p.Scalar(init=np.random.uniform(low=0, high=MAX),
                               lower=0, upper=MAX),
            Kvel = ng.p.Scalar(init=np.random.uniform(low=0, high=MAX),
                               lower=0, upper=MAX),
            Kp = ng.p.Scalar(init=np.random.uniform(low=0, high=MAX),
                             lower=0, upper=MAX),
            Kd = ng.p.Scalar(init=np.random.uniform(low=0, high=MAX),
                             lower=0, upper=MAX),
        )
    if verbose:
        print("done.")
    return instrumentation





if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    # PARAMETERS
    color:str = "yellow"

    params:dict = utils.load_json("/data/params.json")
    
    best = run_bbo(function=obj_function,
            instrumentation=nvg_instrumentation(),
            iterations_per_worker=params['nvg_iterations'],
            optimizer_name=params['optimizer_name'],
            use_NonObjectOptimizer=params['use_NonObjectOptimizer'],
            num_workers=params['n_cores'],
            verbose=params['verbose'],
            target_fitness=params['nvg_target_fitness'],
            MAX_BUDGET=params['nvg_iterations'],
            tolerance=params['nvg_tolerance'],
            time_limit=params['nvg_time_limit'],
            plot_folder=params['nvg_folder'],
            max_possible_fitness=params['nvg_fitness_limit'],
            color=color,
           )
    utils.save_json(best, f"{params['nvg_folder']}/best_params.json")
