## Modello dinamico di un quadricottero

#Import librerie
import numpy as np  
from scipy.integrate import solve_ivp

from suv.trajectory import Traiettoria_Scitech
from suv.controller import Quad_Controller


#Definizione della classe Quadricottero
class Quad:
    
    def __init__(self, params, trajectory=Traiettoria_Scitech()):
        # Parametri fisici del quadricottero
        self.mass = params['mass']  
        self.J_x = params['Inertia']['I_xx']
        self.J_y = params['Inertia']['I_yy']
        self.J_z = params['Inertia']['I_zz']
        self.inertia = np.diag([self.J_x, self.J_y, self.J_z])
        self.l = params['lunghezza_braccio'] 
        self.ang_cross = np.deg2rad(params['angolo_cross'])
        self.l_proj_x = self.l * np.sin(self.ang_cross)
        self.l_proj_y = self.l * np.cos(self.ang_cross)
        self.b = params['b_thrust']
        self.d = params['d_torque']
        self.dob = self.d/self.b  
        
        # Parametri Terra
        self.g = np.array([0, 0, params['gravity']])

        # Traiettoria
        self.trajectory = trajectory
        
    

    def dynamics(self, state, control_inputs):
        
        #Definizione dell'ordine dello stato
        #pos = state[0:3]--> non ci sono forze posizionali
        vel = state[3:6]
        phi = state[6]
        theta = state[7]    
        psi = state[8]
        omega = state[9:12]

        # Eulero in matrice di rotazione 
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        R_BE = np.array([
                        [ctheta * cpsi, ctheta * spsi, -stheta],
                        [sphi * stheta * cpsi - cphi * spsi,  sphi * stheta * spsi + cphi * cpsi,  sphi * ctheta],
                        [cphi * stheta * cpsi + sphi * spsi,  cphi * stheta * spsi - sphi * cpsi,  cphi * ctheta]
                        ])
        ### Prima equazione cardinale ###
        ## Forze ##

        # Attuatori
        thrusts = control_inputs  
        total_thrust = np.array([0, 0, -np.sum(thrusts)])
       
        # Derivata della posizione #
        pos_dot = vel
        # Derivata della velocità #
        vel_dot = (np.transpose(R_BE) @ total_thrust) / self.mass + self.g
    

        ### Seconda equazione cardinale ###
        ## Momenti ##

        # Attuatori
        tau = np.array([
            self.l_proj_x * (-thrusts[0] + thrusts[1] + thrusts[2] - thrusts[3]),
            self.l_proj_y * (+thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3]),
            self.dob * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])  
        ])
        
        # Derivata angoli di Eulero # 
        ang_dot =np.array([
                omega[0]+sphi*stheta/ctheta*omega[1]+cphi*stheta/ctheta*omega[2],
                cphi*omega[1]-sphi*omega[2],
                sphi/ctheta*omega[1]+cphi/ctheta*omega[2]
                ])

        # Derivata della velocità angolare #
        omega_dot = np.linalg.inv(self.inertia) @ (tau - np.cross(omega, self.inertia @ omega))
        
        return np.concatenate((pos_dot, vel_dot, ang_dot, omega_dot))
    

    def run(self,
            data:dict[str,float],
            t_f:float=10.0,
            dt:float=0.01,
            state0:np.ndarray=np.zeros(12),
           ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        '''
        Inizializza il controllore con i parametri dati ed esegue una simulazione in cui il quad segue una traiettoria.
        
        **Arguments**:
        - `data` : dizionario contenente i dati necessari ad inizializzare il controllore
        - `t_f` : tempo finale della simulazione
        - `dt` : passo di integrazione
        - `state0` : Stato iniziale --> [pos(x,y,z), vel(u,v,w), euler(phi,theta,psi), omega(p,q,r)]

        **Returns**:
        - Stato del drone nel tempo:
            - `pos` : posizione
            - `vel` : velocità
            - `quat` : quaternion
            - `omega` : assetto
        '''
        # Creazionedell'istanza del controllore
        controller = Quad_Controller(data, self.trajectory)

        # --- Impostazioni della Simulazione --- #

        # Calcoliamo il numero di passi
        # +1 per includere sia t=0 che t=t_f
        num_steps = int(t_f / dt) + 1

        # Creiamo un array di tutti i punti temporali
        t_punti = np.linspace(0, t_f, num_steps)

        # --- Pre-allocazione della Soluzione ---
        # Otteniamo il numero di stati da state0
        num_stati = state0.shape[0] 

        # Creiamo un array vuoto (12, num_steps) per contenere TUTTI i risultati
        # Questo è molto più efficiente che usare np.append
        soluzione = np.empty((num_stati, num_steps))

        # Salviamo lo stato iniziale nel primo slot (t=0)
        soluzione[:, 0] = state0
        current_state = state0

        # --- Ciclo di Simulazione ---
        # Iteriamo (num_steps - 1) volte, perché abbiamo già lo stato a t=0
        for i in range(num_steps - 1):
            # Definiamo l'intervallo di tempo PER QUESTO PASSO
            t_start = t_punti[i]
            t_end = t_punti[i+1]
            t_span_step = (t_start, t_end)
            
            # Valutiamo la soluzione solo alla fine di questo intervallo
            t_eval_step = [t_end] 

            # 1. Calcolo degli input di controllo
            # Calcolati all'inizio dell'intervallo (t_start) e tenuti costanti
            control_inputs = controller.control(current_state, t_start)
            
            # 2. Risoluzione del modello per un singolo passo
            sol = solve_ivp(
                fun=lambda t, state: self.dynamics(state, control_inputs), 
                t_span=t_span_step, 
                y0=current_state, 
                t_eval=t_eval_step, 
                method='RK45'
            )
            
            # 3. Aggiornamento dello stato per la prossima iterazione
            current_state = sol.y[:, -1]  # Prendiamo l'ultimo punto (l'unico in t_eval)
            
            # 4. Salvataggio dei risultati
            # Inseriamo il nuovo stato nello slot corretto dell'array pre-allocato
            soluzione[:, i + 1] = current_state

        # Alla fine del loop, 'soluzione' conterrà l'intera traiettoria di stato.
        # print(np.shape(soluzione))
        tempo = np.linspace(0, t_f, 1001)
        # print(tempo)
        # Estrazione dei risultati
        pos = soluzione[0:3, :]
        vel = soluzione[3:6, :]
        quat = soluzione[6:10, :]
        omega = soluzione[10:13, :]
        return pos, vel, quat, omega
        
        