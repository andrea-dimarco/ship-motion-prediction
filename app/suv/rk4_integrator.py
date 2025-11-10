def rk4_step(dynamics, state, u, dt):
    # Explicit RK4 integration for state derivative dynamics
    k1 = dynamics(state, u)
    k2 = dynamics(state + 0.5*dt*k1, u)
    k3 = dynamics(state + 0.5*dt*k2, u)
    k4 = dynamics(state + dt*k3, u)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)