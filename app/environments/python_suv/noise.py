from numpy.random import multivariate_normal
from numpy import ndarray

class NOISE():

    def __init__(self, mu:ndarray, cov:ndarray):
        self.__mu:ndarray = mu
        self.__cov:ndarray = cov
        self.equation()

    
    def set_mu(self, mu:ndarray) -> None:
        self.__mu = mu
    def set_cov(self, cov:ndarray) -> None:
        self.__cov = cov
    def set_state(self, mu:ndarray, cov:ndarray) -> None:
        self.set_mu(mu)
        self.set_cov(cov)
    def state(self) -> ndarray:
        return self.__state
    

    def algorithm(self, input:ndarray|None=None) -> None:
        # NOTE: no `algorithm` for this module
        pass


    def equation(self, input:ndarray|None=None) -> None:
        self.__state = multivariate_normal(mean=self.__mu, cov=self.__cov)


    def der(self, input:ndarray|None=None) -> ndarray:
        # NOTE: no `der` for this module
        pass


    def step(self, input:ndarray|None=None) -> None:
        self.equation(input)