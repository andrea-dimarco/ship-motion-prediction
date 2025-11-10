import numpy as np




class UAV():

    def __init__(self, x_0:float, y_0:float, z_0:float, phi_0:float=0.0):
        # NOTE: initial conditions
        self.__state = np.array( [x_0, y_0, z_0, phi_0] )

    
    def set_state(self, new:np.ndarray) -> None:
        self.__state = new
    def state(self) -> np.ndarray:
        return self.__state
    

    def G(self) -> np.ndarray:
        phi:float = self.__state[3]
        g = [
            [np.cos(phi),  np.sin(phi), 0.0, 0.0],
            [-np.sin(phi), np.cos(phi), 0.0, 0.0],
            [0.0,          0.0,         1.0, 0.0],
            [0.0,          0.0,         0.0, 1.0]
        ]
        return np.array(g)
    

    def algorithm(self, input:np.ndarray|None=None) -> None:
        # NOTE: no `algorithm` for this module
        pass


    def equation(self, input:np.ndarray|None=None) -> None:
        # NOTE: no `equation` for this module
        pass


    def der(self, action:np.ndarray, noise:np.ndarray) -> np.ndarray:
        '''
        Returns the derivative of the `state` internal variable
        '''
        return np.matmul(self.G(), action) + noise


    def step(self, action:np.ndarray, noise:np.ndarray) -> None:
        '''
        Evolves the system of `1` step
        '''
        # self.equation()
        self.__state += self.der(action, noise)
        # self.algorithm()
