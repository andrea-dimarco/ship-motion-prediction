from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO, DQN # RL algorithm, many possible options
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # helps for parallelism, wrapper for the environment
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np

from typing import Any, Literal
import utils.utils as utils



def net_arch(hidden_size:int,
             num_layers:int,
             critic_discount:float=0.80,
             other_model:Literal['vf','qf']='vf', # 'vf for A2C, PPO, and TRPO
            ) -> list[dict]:
    '''
    Returns the specification of the net formatted as StableBaselines3 wants

    **Arguments**:
    - `hidden_size` : hidden size for each layer of the **actor** model
    - `num_layers` : number of layers for the **actor** model
    - `critic_discount` : the **critic** will have `int(num_layers*critic_discount)` layers and `int(hidden_size*critic_discount)` hidden size
    - `shared_layers` : how many *shared* layers **actor** and **crtic** have
    - `shared_size` : dimension of each of the *shared* layer
    '''
    assert critic_discount > 0 and critic_discount <= 1
    assert num_layers > 0 and hidden_size > 0 
    return { 'pi' : [int(hidden_size) for _ in range(num_layers)], # actor (policy)
             other_model : [int(hidden_size*critic_discount) for _ in range(int(num_layers*critic_discount))], # critic (value function)
           }
    # print(arch)
    # exit()
    # return arch


class DummyAgent():

    def __init__(self, action_space):
        '''
        Gven a *Gym* `action_space` it returns random actions.
        '''
        self.action_space = action_space

    def action(self, env_state:Any|None=None) -> Any:
        '''
        Returns a *random* action.
        '''
        return self.action_space.sample()
    
    def name(self) -> str:
        return "dummy"



class BasicAgent:

    def __init__(self, *,
                 net_arch:list[dict[str,list[int]]]|None=None,
                 load_path:str|None=None,
                 policy:str|None=None,
                 env:Any|None=None,
                 reward_limit:float|None=None,
                 model_name:str="PPO",
                 verbose:bool=False,
                 log_path:str|None=None,
                 learning_rate:float|None=None,
                 discount_factor:float|None=None,
                ):
        assert not load_path is None or not net_arch is None
        self.__algorithm = PPO
        if reward_limit:
            self.__stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_limit, verbose=int(verbose))
        else:
            self.__stop_callback = None
        if load_path:
            self.load(file_path=load_path, algorithm=PPO, env=env)
        else:
            self.__wrap_env = DummyVecEnv([lambda: env])
            
            if discount_factor is None or learning_rate is None:
                self.__model = self.__algorithm(policy,
                                                self.__wrap_env,
                                                verbose=int(verbose),
                                                tensorboard_log=log_path,
                                                policy_kwargs={'net_arch':net_arch},
                                               )
            else:
                self.__model = self.__algorithm(policy,
                                                self.__wrap_env,
                                                verbose=int(verbose),
                                                tensorboard_log=log_path,
                                                policy_kwargs={'net_arch':net_arch},
                                                gamma=discount_factor,
                                                learning_rate=learning_rate,
                                               )
        if model_name:
            self.set_name(model_name)

    def model(self) -> Any:
        return self.__model
    def set_name(self, new:str) -> None:
        self.__model_name = new
    def name(self) -> str:
        return self.__model_name
    def save(self, folder_path:str) -> None:
        self.__model.save(f"{folder_path}{self.name()}")
    def load(self, file_path:str, env:Any|None=None, algorithm:PPO|None=None) -> None:
        if algorithm is not None:
            self.__algorithm = algorithm
        self.__model = self.__algorithm.load(file_path)
        if env is None:
            self.__model.env = self.__wrap_env
        else:
            self.__wrap_env = DummyVecEnv([lambda: env])
            self.__model.env = self.__wrap_env



    def train(self, time_steps:int, save_folder_path:str|None=None, val_frequency:int=10000, verbose:bool=True) -> None:
        if self.__stop_callback is None:
            self.__model.learn(total_timesteps=time_steps)
        else:
            eval_callback = EvalCallback(
                            self.__wrap_env, # environment
                            callback_on_new_best=self.__stop_callback, # check callback only on new best model
                            eval_freq=val_frequency,
                            best_model_save_path=save_folder_path,
                            verbose=int(verbose)
                        )
            self.__model.learn(total_timesteps=time_steps, callback=eval_callback)
        if save_folder_path:
            self.save(save_folder_path)


    def action(self, env_state:Any) -> Any:
        action, next_state = self.__model.predict(env_state)
        return action



# EXAMPLE OF CUSTOM ENVIRONMENT
class CustomEnvironment(Env):
    '''
    # Gymnasium Custom Environment

    More info can be found [at this link](https://gymnasium.farama.org/api/env/)

    ## Observation & Action spaces
    Gym environments supports different types of environments:
    - **Box**: n dimensional tensor, range of continuous values (e.g. Box(10,50,shape=(3,3)))
    - **Discrete**: set of items (e.g. Discrete(3))
    - **Tuple**: tuple of other spaces (e.g. Tuple(Discrete(2), Box(0,100,shape=(1,))))
    - **Dict**: dictionary of spaces (e.g. Dict({'height':Discrete(2), 'speed':Box(0,100,shape=(1,))}))
    - **MultiBinary**: one-hot encoded binary values (e.g. MultiBinary(4))
    - **MultiDiscrete**: multiple (vector of) discrete values with possibly different ranges (e.g. MultiDiscrete([3,5,2])) 
    '''
    
    def __init__(self, name,
                 action_space:Box|Discrete|Dict|Tuple|MultiBinary|MultiDiscrete,
                 observation_space:Box|Discrete|Dict|Tuple|MultiBinary|MultiDiscrete,
                 initial_state:Any|None=None,
                 max_episode_length:int|None=None,
                 reward_threshold:float|None=None
                ):
        self.__name:str = name
        self.__reward_threshold:float|None = reward_threshold

        self.action_space = action_space
        self.observation_space = observation_space
        if initial_state:
            self.state = initial_state
        else:
            self.state = self.random_state()
        self.max_episode_length = max_episode_length
        if max_episode_length:
            self.current_episode_time = 0
        

    def reward(self, state, action) -> float:
        '''
        Computes and returns the reward for the agent
        '''
        if state >= 45 and state <= 55:
            reward = 1.0
        else:
            reward = 0.0
        return reward
    

    def info(self) -> dict[str,Any]:
        '''
        Auxiliary information to complement the agent observations
        '''
        return None
    

    def termination(self) -> bool:
        '''
        Decides whether the simulation must end or not
        '''
        if self.max_episode_length:
            self.current_episode_time += 1
            if self.current_episode_time >= self.max_episode_length:
                return True
        return False


    def update_state(self, action:Any) -> None:
        '''
        Updates the internal state of the Enironment fron the provided `action`
        '''
        self.state = max(min(self.state + (action-1), 100), 0) # keep state within 0 and 100


    def truncation(self) -> bool:
        # TODO: what should this do?
        return False

    def observe(self) -> np.ndarray:
        '''
        Returns the agent's observation
        '''
        return np.array([self.state])

    def step(self, action:Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        '''
        Performs an action, updates the state and returns tuple (`observation`, `reward`, `terminated`, `truncated`, `info`)
        tion to the state, the goal is to keep the state between 45 and 55
        This example adds the action

        **Returns**: (`observation`, `reward`, `terminated`, `truncated`, `info`)
        '''
        # transition state
        self.update_state(action=action)

        return self.observe(), self.reward(state=self.state, action=action), self.termination(), self.truncation(), self.info()

    
    def render(self):
        # not doing it as of now
        pass


    def reset(self, seed:int|None=None, options:dict|None=None) -> tuple[np.ndarray, dict[str, Any]]:
        '''
        Resets the state of the sistem to a (possibly random) initial state

        **Returns**: (`observation`, `info`)
        '''
        super().reset(seed=seed) # NOTE: necessary for custom envs
        self.current_episode_time = 0
        self.state = self.random_state()

        return self.observe(), self.info()


    def reward_limit(self) -> float|None:
        return self.__reward_threshold
    def name(self) -> str:
        return self.__name
    def random_action(self) -> Any:
        return self.action_space.sample()
    def random_state(self) -> Any:
        return self.observation_space.sample()


    def run(self, episodes:int=5, agent:Any|None=None, verbose:bool=True, do_render:bool=False):
        '''
        Executes `episodes` episodes and prints some useful information
        '''
        if verbose:
            print(f"Observation space is:", end=" ")
            utils.print_colored(str(self.observation_space), color="blue")
            print(f"Action space is:", end=" ")
            utils.print_colored(str(self.action_space), color="blue")
            print("Performing test:")
            bar = utils.BAR(episodes, start_immediately=True)
        score_history:list[float] = list()
        episode_length_history:list[int] = list()
        for _ in range(episodes):
            observation, _ = self.reset()
            done:bool = False
            score:float = 0.0
            episode_length:int = 0
            while not done:
                if do_render:
                    self.env().render() # graphical representation of environment
                if agent:
                    action = agent.action(observation)
                else:
                    action = self.random_action()
                observation, reward, done, truncated, info = self.step(action)
                score += reward
                episode_length += 1
            if verbose:
                episode_length_history.append(episode_length)
                score_history.append(score)
                bar.update()
        if verbose:
            print(f"\tAvg score:", end=" ")
            utils.print_colored(sum(score_history)/len(score_history), color="green")
            print(f"\tAvg episode length:", end=" ")
            utils.print_colored(sum(episode_length_history)/len(episode_length_history), color="green")
        self.close()



def simple_evaluation(agent, env, eval_episodes:int, verbose:bool=True) -> tuple[float,float]:
    '''
    Perform a simple evaluation of the policy and computes average returned score an standard deviation.

    **Return**: (`avg_score`, `std_score`)
    '''
    from stable_baselines3.common.evaluation import evaluate_policy
    # EVALUATION
    if verbose:
        utils.print_colored("Evaluation", highlight="light_yellow")
    avg_score, std_score = evaluate_policy(agent,
                                           env,
                                           n_eval_episodes=eval_episodes,
                                           render=False
                                          )
    env.close()
    if verbose:
        print("Trained model has:\n\tAvg Score:", end=" ")
        utils.print_colored(avg_score, color="light_yellow")
        print("\tStd Score:", end=" ")
        utils.print_colored(std_score, color="light_yellow")
    return avg_score, std_score



def format_env(env, num_cpus:int=1, variable_num_agents:bool=False, verbose:bool=True) -> Any:
    '''
    Given a PettingZoo environment it changes it to a Gym-type (StableBaselines3-compatible) environment.

    The environment can be multi-agent but each agent has to be represented by the **same policy** (one deep model that governs all agents).
    
    Note that the action space and observation space have to be the same amongst **all agents**, *rewards* and individual *observations* can differ in value (but **not in type**).
    '''
    import supersuit as ss
    # Wrap with SuperSuit to pad observations & actions
    new_env = ss.pettingzoo_env_to_vec_env_v1(env)
    new_env.black_death = variable_num_agents # to support changing number of agents
    new_env = ss.concat_vec_envs_v1(new_env, num_vec_envs=1, num_cpus=num_cpus, base_class="stable_baselines3")

    return new_env