import numpy as np

from environments.python_suv.agent import AGENT
from environments.python_suv.noise import NOISE
from environments.python_suv.uav import UAV



class SYSTEM():

    def __init__(self,
                 v_max:float,
                 omega_max:float,
                 initial_position:list|tuple|np.ndarray,
                 mu:np.ndarray,
                 cov:np.ndarray,
                ) -> None:
        self.__agent:AGENT = AGENT(v_max=v_max,omega_max=omega_max)
        self.__uav:UAV = UAV(x_0=initial_position[0],
                              y_0=initial_position[1],
                              z_0=initial_position[2],
                              phi_0=initial_position[3],
                             )
        self.__noise:NOISE = NOISE(mu=mu, cov=cov)


    def set_initial_state(self, initial_position:list|tuple|np.ndarray) -> None:
        self.__uav.set_state(initial_position)

    def state(self) -> dict[str,np.ndarray]:
        return self.__uav.state()

    def step(self, action:np.ndarray) -> None:
        '''
        Evolves the system of `1` step
        '''
        self.__noise.step()
        self.__agent.step(action)
        self.__uav.step(action=self.__agent.state(), noise=self.__noise.state())