import numpy as np




class AGENT():

    def __init__(self, v_max:float, omega_max:float, v_hat_x_0:float=0.0, v_hat_y_0:float=0.0, v_hat_z_0:float=0.0, omega_hat_z_0:float=0.0):
        # NOTE: initial conditions
        self.__state:np.ndarray = np.array( [v_hat_x_0, v_hat_y_0, v_hat_z_0, omega_hat_z_0] )
        self.__saturation = np.array( [v_max, v_max, v_max, omega_max] )


    def set_state(self, new:np.ndarray) -> None:
        self.__state = new
    def state(self) -> np.ndarray:
        return self.__state
    

    def algorithm(self, input:np.ndarray|None=None) -> None:
        # NOTE: no `algorithm` for this module
        pass


    def equation(self, input:np.ndarray|None=None) -> None:
        self.__state = np.multiply(input, self.__saturation)


    def der(self, input:np.ndarray|None=None) -> np.ndarray:
        # NOTE: no `der` for this module
        pass


    def step(self, input:np.ndarray) -> None:
        '''
        Evolves the system of `1` step
        '''
        self.equation(input)