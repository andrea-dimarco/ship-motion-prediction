## Controllore PD quadricottero

# Import librerie
import numpy as np  

# Definizione della classe Quadricottero
class Quad_Controller:
    def __init__(self, params,trajectory):
        # Guadagni del controllore PD
        self.Kpos = params['Kpos']  
        self.Kvel = params['Kvel']
        self.Kp = params['Kp']
        self.Kd = params['Kd']
        self.psi_des = params['psi_des']
        self.omega_des = np.array([0, 0, 0])
        self.limite_zddot = params['limite_zddot']
        self.T_c_max = params['Coeff trazione massima']*params['mass']*params['gravity']
        self.limite_ang = params['limite_ang']
        self.banda_T = params['banda_T']
        # Grandezze drone e terra
        self.m = params['mass']
        self.g = params['gravity']
        self.l = params['lunghezza_braccio'] 
        self.ang_cross = np.deg2rad(params['angolo_cross'])
        self.l_proj_x = self.l * np.sin(self.ang_cross)
        self.l_proj_y = self.l * np.cos(self.ang_cross)
        self.b = params['b_thrust']
        self.d = params['d_torque']
        self.dob = self.d/self.b  
        #Traiettoria di riferimento
        self.trajectory = trajectory

    def control(self, state, time):
        # Definizione dell'ordine dello stato
        pos = state[0:3]
        vel = state[3:6]
        ang = state[6:9]
        omega = state[9:12]

        # Definizione della reference
        reference = self.trajectory.TrajectoryEval(time)
        pos_ref = reference[0:3]
        vel_ref = reference[3:6]

        # Calcolo dell'errore
        pos_error = pos_ref - pos
        vel_error = vel_ref - vel

        # Accelerazioni desiderate
        acc_des = self.Kpos * pos_error + self.Kvel * vel_error

        # Calcolo trazione e degli angoli desiderati 
        acc_z_des = -np.clip(acc_des[2] - self.g, -self.T_c_max, -self.limite_zddot) 
        acc_y_des = np.clip(acc_des[1], -acc_z_des*np.tan(self.limite_ang), acc_z_des*np.tan(self.limite_ang))
        acc_x_des = np.clip(acc_des[0], -self.T_c_max, self.T_c_max)
        T_des = np.clip(self.m*np.sqrt(acc_x_des**2 + acc_y_des**2 + acc_z_des**2),0,self.T_c_max)
        phi_des = np.asin(np.clip(acc_y_des / (T_des/self.m),-1,1))
        theta_des = 2*np.atan(-acc_x_des/(np.cos(phi_des)*(T_des/self.m)+acc_z_des))
        
        # Calcolo dei momenti desiderati
        ang_des = np.array([phi_des, theta_des, self.psi_des])
        ang_error = ang_des - ang
        omega_error = self.omega_des - omega
        tau_des = self.Kp * ang_error + self.Kd * omega_error   
        
        # Conversione in singole trazioni
        delta_tau_phi = tau_des[0]/(4*self.l_proj_x)
        delta_tau_theta = tau_des[1]/(4*self.l_proj_y)
        delta_tau_psi = tau_des[2]/(4*self.dob)

        T_1 = np.clip((T_des/4) - delta_tau_phi + delta_tau_theta + delta_tau_psi,0,self.T_c_max/4)
        T_2 = np.clip((T_des/4) + delta_tau_phi + delta_tau_theta - delta_tau_psi,0,self.T_c_max/4)
        T_3 = np.clip((T_des/4) + delta_tau_phi - delta_tau_theta + delta_tau_psi,0,self.T_c_max/4)
        T_4 = np.clip((T_des/4) - delta_tau_phi - delta_tau_theta - delta_tau_psi,0,self.T_c_max/4)

        return np.array([T_1, T_2, T_3, T_4])