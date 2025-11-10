import numpy as np

## Traiettoria del quadricottero
class Traiettoria_Scitech:
    def __init__(self):
        self.tf = 10
        #self.time = np.arange(0, tf + 0.01, 0.01)
        self.dy_ramp = 3
        self.Vmax = 3              
        self.R = 1
        self.Tx = self.tf/2 
        self.Rz = 7
        self.Tz = self.tf

    def TrajectoryEval(self, time) -> np.ndarray:
        '''
        Returns the trajectory state at the given time

        **Arguments**:
        - `time` : at which time the trajectory must be sampled from

        **Returns**:
        - `xr` : trajectory state in the form
            - `x` : coordinate on x axis
            - `y` : coordinate on y axis
            - `z` : coordinate on z axis
            - `u` : velocity on x axis
            - `v` : velocity on y axis
            - `w` : velocity on z axis
        '''
        if time <= self.dy_ramp:
            ydot = self.Vmax/2 * (1 - np.cos(np.pi * (time) / self.dy_ramp))
            y = self.Vmax/2 * (time - (self.dy_ramp/np.pi)*np.sin(np.pi * (time) / self.dy_ramp))
        elif time > self.dy_ramp:
            ydot= self.Vmax
            y = self.Vmax * (time - self.dy_ramp)+self.Vmax/2 *self.dy_ramp 
        x = self.R*np.cos(2*np.pi/self.Tx * time) - self.R
        xdot = -self.R*2*np.pi/self.Tx*np.sin(2*np.pi/self.Tx * time)

        z = self.Rz*np.cos(2*np.pi/self.Tz*time) - self.Rz
        zdot = -self.Rz*2*np.pi/self.Tz*np.sin(2*np.pi/self.Tz * time)

        # Vettore traiettoria
        xr = np.array([x,y,z,xdot,ydot,zdot])
        # position and velocity
        return xr