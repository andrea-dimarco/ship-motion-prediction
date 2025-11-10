import time 
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from typing import Literal
import matplotlib.pyplot as plt
from typing import Any
from pickle import dump, load




class BAR():

    def __init__(self, length:int, start_immediately:bool=True):
        self.reset(new_length=length, start_immediately=start_immediately)
    
    def start(self) -> None:
        self.__initial_timer.start()
        self.__timer.start()
        if self.__initial_timer_took:
            self.__bar.start()
        else:
            self.__k:int = 0
    
    def update(self, increment:int=1, status:int=None, auto_finish:bool=True) -> None:
        # compute new status
        if status:
            self.set_status(status)
        else:
            self.set_status(self.status() + increment)
        if not self.__initial_timer_took:
            self.__initial_timer.end()
            self.__initial_timer_took = True
        if not self.__has_started:
            print(f"Expected duration: {TimeExecution.seconds_to_string( (self.__initial_timer.elapsed()/self.status())*(self._get_finish_line()-self.status()) )}")
            self.__has_started = True
            self.start()
        # update bar
        if auto_finish and (self.status() >= self._get_finish_line()):
            self.end()
        else:
            self.__bar.update(self.status())
    
    def set_status(self, new:int) -> None:
        self.__k = new

    def status(self) -> int:
        return self.__k
    
    def _get_finish_line(self) -> int:
        return self.__max_k
    
    def is_done(self) -> bool:
        return self.__is_finished

    def end(self) -> None:
        if not self.is_done():
            self.__bar.finish()
            self.__is_finished = True
            self.__timer.end()
            self.__timer.print()
    
    def finish(self) -> None:
        self.end()

    def reset(self, new_length:int, start_immediately:bool=True) -> None:
        import progressbar
        self.__bar = progressbar.ProgressBar(maxval=new_length,
                                      widgets=[progressbar.Bar('#', '[', ']'),
                                               '.',
                                               progressbar.Percentage()
                                              ]
                                     )
        self.__is_finished:bool = False
        self.__k:int = 0
        self.__max_k:int = new_length
        self.__timer:TimeExecution = TimeExecution()
        self.__initial_timer:TimeExecution = TimeExecution()
        self.__initial_timer_took:bool = False
        self.__has_started:bool = False
        if start_immediately:
            self.start()



def set_seed(seed=0) -> None:
    """
    Sets the seed of the stochastic modules
    """
    try:
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        pass



def str_to_datetime(data:str,
                    data_format:str="%Y-%m-%d %H:%M:%S"
                    ) -> datetime.datetime:
    '''
    Given a string following the given format into a date and returns the month.

    Arguments:
        - data: the date as a string.
        - data_format: the format string the data must follow for this function to work.
    '''
    return datetime.datetime.strptime(data, data_format)



def delete_file(file_path:str) -> None:
    '''
    Deletes the file. 
    '''
    if type(file_path) == str:
        import os
        os.remove(file_path)



def min_max_scaling(value:float, min:float, max:float) -> float:
    '''
    Scales a given **positive** value in a [0,1] range with respect to min and max values, like so:
        return (value - min) / (max - min)
    '''
    assert value >= 0.0, f"value ({value}) must be non-negative"
    return (value - min) / (max - min)


def min_max_scaling_sign(value:float, abs_min:float, abs_max:float) -> float:
    assert abs_min >= 0 and abs_max > 0
    return min_max_scaling(abs(value), abs_min, abs_max) * (-1 if value < 0 else 1)


def z_score_norm(value:float, mu:float, std:float) -> float:
    '''
    Returns the value scaled according to the z-score normalization
    '''
    return (value-mu)/std



def save_obj(obj:Any, file_path:str, verbose:bool=False) -> None:
    if verbose:
        print(f"Saving object to '{file_path}' ... ", end="")
    with open(file_path, 'wb') as fp:
        dump(obj, fp)
    if verbose:
        print("done.")

def load_obj(file_path:str, verbose:bool=False) -> Any:
    if verbose:
        print(f"Loading object from '{file_path}' ... ", end="")
    with open (file_path, 'rb') as fp:
        obj = load(fp)
    if verbose:
        print("done.")
    return obj



def save_json(dictionary:dict, file_path:str) -> None:
    try:
        import commentjson as json
    except ModuleNotFoundError:
        import json
    with open(file_path, 'w') as f:
        json.dump(dictionary, f)

def load_json(file_path:str) -> dict:
    try:
        import commentjson as json
    except ModuleNotFoundError:
        import json
    with open(file_path) as f:
        return json.load(f)
    
def load_yaml(file_path:str) -> dict:
    import yaml
    with open(file_path, "r") as f:
        params = yaml.safe_load(f)
    return params

def save_yaml(dictionary:dict, file_path:str) -> dict:
    import yaml
    with open(file_path, "w") as f:
        yaml.safe_dump(data=dictionary, stream=f)


def get_files_in_dir(directory:str) -> list[str]:
    import os
    all_files:list[str] = list()
    for file_obj in os.scandir(path=directory):
        file_obj.name
        if file_obj.is_file():
            all_files.append(file_obj.name)
    return all_files


def discretize(value:float, step:float=0.01, place_in_middle:bool=False) -> float:
    '''
    Returns another value in its discretized form, by only allowing a mesh over the real space.
    Mesh granularity depends on the `step` argument.
    '''
    try:
        if np.isnan(value):
            return value
    except TypeError:
        return value
    return int(value/step)*step + (0 if not place_in_middle else step/2)



class TimeExecution():
    
    def __init__(self):
        self.__start_time = None
        self.__end_time = None
    
    def elapsed(self):
        assert(not (self.__start_time is None))
        try:
            return self.__end_time - self.__start_time
        except TypeError:
            return time.time() - self.__start_time
    
    def start(self) -> None:
        self.__start_time = time.time()
        self.__end_time = None
    
    def end(self) -> None:
        assert(not (self.__start_time is None))
        self.__end_time = time.time()

    def print(self, indent:bool=True) -> str:
        if indent:
            print(f"\tExecution took {str(self)}")
        else:
            print(f"Execution took {str(self)}")

    def seconds_to_string(seconds_total:int) -> str:
        minutes:int = int(seconds_total // 60)
        hours:int = int(minutes // 60)
        minutes = int(minutes % 60)
        seconds:int = round(seconds_total % 60, 2)
        string = f"{hours}h {minutes}m {seconds}s"
        return string
        
    def __str__(self):
        '''
        If `silent` it returns the string without printing it.
        '''
        return TimeExecution.seconds_to_string(seconds_total=self.elapsed())

    

def scan_dir(folder_path:str) -> tuple[list[str], list[str]]:
    '''
    Returns tuple (`files_found`, `folders_found`)
    '''
    import os
    folders_found:list[str] = list()
    files_found:list[str] = list()

    for obj in os.scandir(path=folder_path):
        if obj.is_dir():
            folders_found.append(obj.name)
        elif obj.is_file():
            files_found.append(obj.name)
    return files_found, folders_found


def print_colored(text, color:str="black", highlight:str|None=None, end:str='\n') -> None:
    '''
    Supported colors:
    - black	
    - red	
    - green	
    - yellow	
    - blue	
    - magenta	
    - cyan	
    - white	
    - light_grey	
    - dark_grey	
    - light_red	
    - light_green	
    - light_yellow	
    - light_blue	
    - light_magenta	
    - light_cyan

    Highlighted text:
    - on_{color}
    '''
    from termcolor import colored
    if (highlight is None) or ("on_" in highlight):
        print(colored(str(text), color, highlight), end=end)
    else:
        print(colored(str(text), color, f"on_{highlight}"), end=end)



def elapsed_months(date1:datetime.datetime, date2:datetime.datetime, days_in_month:int=31) -> float:
    '''
    Returns the number of months between two dates.
    If `date1` is later than `date2`, it returns a negative number.
    '''
    # Ensure date1 is the earlier date
    change_sign:bool = False
    if date1 > date2:
        change_sign = True
        date1, date2 = date2, date1
    
    # Calculate year and month difference
    months = (date2 - date1).days / days_in_month

    return months if not change_sign else -months



def elapsed_years(start:datetime.datetime|str, end:datetime.datetime|str, format:str="%Y-%m-%d") -> int:
    if isinstance(start, str):
        start = datetime.datetime.strptime(start, format)
    if isinstance(end, str):
        end = datetime.datetime.strptime(start, format)
    return int((end - start).days / 365.25)  # Approximate, accounts for leap years


from concurrent import futures
def embarassing_parallelism(function, n_workers:int, arguments_list:list, use_process:bool=False, use_kwargs:bool=False) -> list:
    '''
    Launches `n_workers` parallel jobs, with no communication between them _(embarassing parallelisms)_ and returns the **unordered** iterable list with their results.
    Each job `i` executes `function` with input `arguments_list[i]`, we recommend the input to be a dictionary.
    
    Returns the iterable `results` with each job's results, we recommend the iterable to return a dictionary.

    ### Arguments ###
    - `function`: the executable to be ran
    - `n_workers`: number of parallel jobs to launch
    - `arguments_list`: list of inputs for the jobs _(see above description)_
    - `use_process`: if `True` it launches `n_workers` **processes**, else it launches `n_workers` **threads**
    '''
    if len(arguments_list) != n_workers:
        raise IndexError(f"Length of 'argument_list' (now {len(arguments_list)}) must be equal to 'n_workers' (now {n_workers}).")
    if not use_kwargs:
        if n_workers > 1:
            if use_process:
                with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(function,
                                                arguments_list,
                                            )
                                )
            else:
                with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(function,
                                                arguments_list,
                                            )
                                )
            return results
        else:
            return [function(arguments_list[0])]
    else:
        if n_workers > 1:
            if use_process:
                with futures.ProcessPoolExecutor() as executor:
                    Fs = [executor.submit(function, **kwargs) 
                               for _, kwargs in enumerate(arguments_list)
                              ]
                    results = [f.result() for f in Fs]
            else:
                with futures.ThreadPoolExecutor() as executor:
                    Fs = [executor.submit(function, **kwargs) 
                               for _, kwargs in enumerate(arguments_list)
                              ]
                    results = [f.result() for f in Fs]
            return results
        else:
            return [function(**arguments_list[0])]

     

def most_common_element(lst:list[Any]):
    return max(set(lst), key=lst.count)


def print_dict(d:dict, indent=0, key_color:str="yellow", key_highlight:str|None=None, value_color:str="black", value_highlight:str|None=None) -> None:
    for key, value in d.items():
        print('\t' * indent, end="")
        print_colored(str(key), color=key_color, highlight=key_highlight, end=": ")
        if isinstance(value, dict):
            print('')
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1), end="")
            print_colored(str(value), color=value_color, highlight=value_highlight, end="\n")


def invert_dict(d:dict) -> dict:
    return {v: k for k, v in d.items()}


def get_object_methods(obj:Any) -> set[str]:
    '''
    Returns a list of the callable methods of an object.
    '''
    import inspect
    return {name for name, member in inspect.getmembers(obj, predicate=inspect.ismethod)}


def get_method_info(obj:Any, method_name:str, verbose:bool=False) -> tuple[str, str]:
    '''
    This function returns the signature and (if present) the documentation of a method `method_name` for the object `obj`.
    '''
    import inspect
    try:
        method = getattr(obj, method_name)
        signature = inspect.signature(method)
        docstring = method.__doc__
    except AttributeError:
        raise AttributeError(f"The object does not have a method named '{method_name}'.")
    except ValueError:
        raise ValueError(f"Cannot retrieve the signature for the method '{method_name}'.")
    if verbose:
        print_colored("Signature:", color="yellow", end="\n")
        print(signature)
        print_colored("Documentation:", color="yellow", end="\n")
        print(docstring)
    return signature, docstring


def get_object_info(obj:Any, verbose:bool=False, title_color:str="yellow", text_color:str="white", value_color:str="dark_grey") -> tuple[dict[str], set[str]]:
    """
    Returns a tuple `attributes:dict` (a dict with attribute names and values) and `methods:set` (a set with methods names).

    To get more information about specific methods use `get_method_info(obj,method_name,verbose)`
    """
    try:
        attributes:dict[str,Any] = vars(obj)
    except TypeError:
        import inspect
        attributes:list = inspect.getmembers(obj, lambda a: not(inspect.isroutine(a)))
    methods:set[str] = get_object_methods(obj=obj)

    if verbose:
        print_colored("Object Type:", color=title_color, end=" ")
        print(type(obj))
        print_colored("Object Attributes:", color=title_color, end="\n")
        if isinstance(attributes, dict):
            print_dict(attributes, key_color=text_color, value_color=value_color)
        else:
            print_two_column(sorted(attributes), color=text_color)
        print_colored("Object Methods:", color=title_color, end="\n")
        print_two_column(sorted(list(methods)), color=text_color)

    return attributes, methods


def print_two_column(items:list[Any], padding:int=4, color:str="black", hghlight:str|None=None) -> None:
    """
    Prints the items of a list in two columns.

    Parameters:
    - items: List of strings to be printed.
    - padding: Number of spaces between columns (default is 4).
    """
    string:str = ""
    if len(items) > 0:
        # Calculate the number of rows needed,
        half = (len(items) + 1) // 2

        # Split the list into two halves
        left_column = items[:half]
        right_column = items[half:]

        # Determine the width of the left column for alignment
        left_width = max(len(str(item)) for item in left_column) + padding

        # Print the items in two columns
        for i in range(half):
            left_item = str(left_column[i])
            right_item = str(right_column[i]) if i < len(right_column) else ''
            print_colored(f"{left_item:<{left_width}}{right_item}", color=color, highlight=hghlight)
    else:
        print("")




def pad_zeros(value:int, max_value:int) -> str:
    '''
    Pads the value with zeros to match the number of digits in max_value.
    '''
    pad:str = ""
    digits = int(np.log10(max_value))
    try:
        for _ in range(digits - int(np.log10(value))):
            pad += '0'
    except:
        for _ in range(digits):
            pad += '0'
    return f"{pad}{value}"




from PIL import Image
import os

# Settings
def png_to_gif(input_folder:str, output_gif:str, 
               duration:int=200,
               n_loops:int=0,
               delete_pngs:bool=False,
               verbose:bool=False,
              ) -> None:
    """
    - `duration` how many *ms* a single frame should last
    - `n_loops` amount of loops the .gif has to do, `n_loops=0` will loop forever
    - `delete_pngs` where to delete the original .png files in the folder
    """
    # Get all .png files in the folder, sorted by name
    image_files = sorted([
        file for file in os.listdir(input_folder) if file.endswith('.png')
    ])
    # Load images
    images = [Image.open(os.path.join(input_folder, file)) for file in image_files]
    # Save as GIF
    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=n_loops,
        )
        if verbose:
            print(f"GIF saved as '{output_gif}'")
    else:
        if verbose:
            print("No .png images found in the folder.")
    if delete_pngs:
        if verbose:
            print("Deleting original .png files")
        for f in image_files:
            delete_file(os.path.join(input_folder, f))



def png_to_mp4(input_folder:str,
               output_video:str,
               fps:int=24,
               last_frame_copies:int=2,
               delete_pngs:bool=False,
               verbose:bool=False,
              ) -> None:
    import shutil
    import ffmpeg

    # Specify the folder containing the PNG images
    if not delete_pngs:
        old_raw_path = input_folder + "/old_raw"
        try:
            make_directory(old_raw_path, verbose=verbose)
        except FileExistsError:
            pass

    # List all PNG files in the folder and get the last frame
    png_files = sorted([
            file for file in os.listdir(input_folder) if file.endswith('.png')
        ])
    if not png_files:
        raise ValueError("No PNG files found in the folder.")

    last_frame = png_files[-1]

    # Create temporary copies of the last frame to extend its duration
    temp_frames_folder = input_folder + "/temp_frames"
    try:
        make_directory(temp_frames_folder, verbose=verbose)
    except FileExistsError:
        pass

    for frame in png_files:
        shutil.copy(f"{input_folder}/{frame}", f"{temp_frames_folder}/{frame}")  # Copy all original frames to temp

    for i in range(last_frame_copies):  # Add `last_frame_copies` copies of the last frame
        temp_frame_name = f"last_frame_copy_{i:02d}.png"
        shutil.copy(f"{input_folder}/{last_frame}", f"{temp_frames_folder}/{temp_frame_name}")

    # Create the video using the temp folder
    input_pattern = str(temp_frames_folder + "/*.png")
    ffmpeg.input(input_pattern, pattern_type="glob", framerate=fps) \
        .output(output_video, c="libx264", pix_fmt="yuv420p") \
        .overwrite_output() \
        .run(capture_stdout=True, capture_stderr=True)

    if verbose:
        print(f"Video created successfully: '{output_video}'")

    # Clean up temporary frames
    shutil.rmtree(temp_frames_folder)

    # Move original PNG images to old_raw folder
    # for frame in png_files:
    #     shutil.move(str(frame), old_raw_path / frame.name)
    # if verbose:
    #     print(f"All original PNG images moved to: {old_raw_path}")

    if delete_pngs:
        if verbose:
            print("Deleting original .png files")
        for f in png_files:
            delete_file(os.path.join(input_folder, f))



def make_directory(name:str, verbose:bool=False) -> None:
    """
    Creates the directory (and even the nested directories)
    """
    import os
    os.makedirs(name)
    if verbose:
        print(f"Directory '{name}' created successfully.")


def list_n_steps(number:float|int, steps:int) -> list[int]:
    return [(i+1)*(number/steps) for i in range(steps)]


def increasing_exponential(input:float, max_value:float, scaling_factor:float=1.0) -> float:
    """
    A function that **increases** as `input` grows with `max_value` as an asimptote.

    `input` must be non-negative.
    """
    assert input >= 0.0
    return float( max_value*(1 - 1/np.exp(input/scaling_factor)) )


def decreasing_exponential(input:float, max_value:float, scaling_factor:float=1.0) -> float:
    """
    A function that **decreases** as `input` grows with `max_value` as a starting point.

    `input` must be non-negative.
    """
    assert input >= 0.0
    return float( max_value/np.exp(input/scaling_factor) )


def euclidean_distance(position_1:tuple[float,float,float], position_2:tuple[float,float,float]) -> float:
    """
    Computes the euclidean distance between two points in space
    """
    x1, y1, z1 = position_1
    x2, y2, z2 = position_2
    return float( np.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 ) )


def vector_modulo(vector:list|tuple) -> float:
    """
    Returns the modulo of the vector
    """
    modulo:float = 0.0
    for i in vector:
        modulo += i**2
    return float( np.sqrt(modulo) )


def vector_angle_3D(vector:np.ndarray) -> tuple[float,float]:
    """
    Given a vector in **R^3** returns the angles (`phi`,`theta`) where:
    - `theta`: angle with the `z` axis
    - `phi`: angle with the `x` axis
    """
    phi = np.arctan(vector[1]/vector[0])
    theta = np.arccos(vector[2]/np.linalg.norm(vector))
    return phi, theta


def vector_angle_2D(vector:np.ndarray) -> float:
    """
    Given a vector in **R^2** returns the angle `theta` with the `x` axis
    """
    return np.arctan(vector[1]/vector[0])


def vector_angle_between(v1:np.ndarray, v2:np.ndarray) -> float:
    """
    Returns the angle `theta` between two vectors 'v1` and `v2`
    """
    cosine = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    # clip value due to precision errors
    cosine = clip_value(cosine, -1.0, 1.0)
    return np.arccos( cosine )


def clip_value(value:float, minimum:float, maximum:float) -> float:
    return max(min(value,maximum), minimum )


def rotate_vector_3D(vector:np.ndarray, phi:float, theta:float) -> np.ndarray:
    theta_0, phi_0 = vector_angle_3D(vector)
    radius = np.linalg.norm(vector)
    return spherical_coordinates(radius=radius, theta=theta_0+theta, phi=phi_0+phi)


def spherical_coordinates(radius:float, theta:float, phi:float) -> np.ndarray:
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    return np.array( (x,y,z) )


def circle_radius(area:float) -> float:
    """
    Given a value for the area of a circle, it returns the radius of the circle
    """
    return np.sqrt( abs(area)/np.pi )


def check_sphere_line_intersection(sphere_center:np.ndarray,
                                   sphere_radius:float,
                                   line_origin:np.ndarray,
                                   line_end:np.ndarray,
                                  ) -> bool:
    """
    Returns **true** if the given *line* intersects with the given *sphere*

    Taken shamelessly from [the web](https://paulbourke.net/geometry/circlesphere/index.html#linesphere)

    **Arguments**:
    - `sphere_center` : 3D coordinates of the sphere's center
    - `sphere_radius` : the radius of the sphere
    - `line_origin` : 3D coordinates of the line's origin
    - `line_end` : 3D coordinates of the line's origin
    """
    x1 = line_origin[0]
    y1 = line_origin[1]
    z1 = line_origin[2]

    x2 = line_end[0]
    y2 = line_end[1]
    z2 = line_end[2]

    cx = sphere_center[0]
    cy = sphere_center[1]
    cz = sphere_center[2]

    # d = P2 − P1
    d = line_end - line_origin
    # dx = x2 - x1
    # dy = y2 - y1
    # dz = z2 - z1

    # f = P1 − C
    f = line_origin - sphere_center
    # fx = x1 - cx
    # fy = y1 - cy
    # fz = z1 - cz

    a = np.linalg.norm(d)**2
    b = 2 * np.dot(f,d)
    c = np.linalg.norm(f)**2 - sphere_radius**2

    # Check if either endpoint is inside sphere
    # (segment intersects if one endpoint is inside)
    if np.linalg.norm(f)**2 <= sphere_radius**2:
        return True
    # For second endpoint:
    g = line_end - sphere_center
    if np.linalg.norm(g)**2 <= sphere_radius**2:
        return True
    # Solve quadratic
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        # no real roots: line misses sphere
        return False
    # compute the two roots
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    # Check if either t1 or t2 is within [0,1]
    if (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0):
        return True
    # otherwise: line intersects sphere but the segment does not
    return False


def pretty_number(n:float, sep:str=".") -> str:
    """
    Returns a string that's better readable for humans.
    """
    digits:str = [c for c in str(n - int(n))]
    decimals:str = ""
    if n - int(n) != 0: # no decimal part
        for i in range(len(digits)):
            if i == 0:
                continue
            decimals += digits[i]
    digits = list(reversed([c for c in str(int(n))]))
    chars:list[str] = []
    k:int = 0
    for i in range(len(digits)):
        if k % 3 == 0 and k != 0:
            chars.append(sep)
        if digits[i] != '-':
            chars.append(digits[i])
        k += 1
    integer = ""
    for c in reversed(chars):
        integer += c
    return integer + decimals


def number_sign(number:float, magnitude:float=1.0) -> float:
    if number == 0.0:
        return 0.0
    elif number > 0.0:
        return magnitude
    elif number < 0.0:
        return -magnitude
    else:
        raise ValueError





