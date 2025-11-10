import yaml
import numpy as np
from scipy.integrate import solve_ivp
from suv.quad import Quad  
from suv.controller import Quad_Controller
from suv.trajectory import Traiettoria_Scitech
import matplotlib.pyplot as plt


# Dati quadricottero da file YAML
with open("Data_Quad.yaml", "r") as f:
    data_quad = yaml.safe_load(f)

# Creazione dell'istanza del quadricottero
quad = Quad(data_quad)

# Stato iniziale: [pos(x,y,z), vel(u,v,w), euler(phi,theta,psi), omega(p,q,r)]
state0 = np.zeros(12)

# Creazionedell'istanza del controllore e della traiettoria
trajectory = Traiettoria_Scitech()
controller = Quad_Controller(data_quad, trajectory)

# --- Impostazioni della Simulazione ---
t_f = 10.0       # Tempo finale
dt = 0.01      # Passo di integrazione (e di controllo)

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
        fun=lambda t, state: quad.dynamics(state, control_inputs), 
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
#print(np.shape(soluzione))
tempo = np.linspace(0, t_f, 1001)
#print(tempo)
# Estrazione dei risultati
pos = soluzione[6, :]
# vel = sol.y[3:6, :]
# quat = sol.y[6:10, :]
# omega = sol.y[10:13, :]
plt.figure()
plt.plot(tempo, pos, label='x')
plt.show()
# # Stampa risultato finale
# print("Posizione finale:", pos[:, -1])
# print("Velocità finale:", vel[:, -1])
