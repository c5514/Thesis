import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

ħ = 1.0
m = 1.0
p0 = 5.0
σ_p = 1.0
x0 = 0.0

t_min = 0.0
t_max = 12.0
num_times = 500
t_vals = np.linspace(t_min, t_max, num_times)

Nx = 1000
x_min = -5.0
x_max = 80.0
x = np.linspace(x_min, x_max, Nx)

Np = 2000
p_min = p0 - 10 * σ_p
p_max = p0 + 10 * σ_p
p = np.linspace(p_min, p_max, Np)
dp = p[1] - p[0]

f_p = (1.0 / ((2 * np.pi)**0.25 * np.sqrt(σ_p))) * np.exp(-(p - p0)**2 / (4 * σ_p**2))

prefactor = 1.0 / np.sqrt(2 * np.pi)
M = np.exp(1j * np.outer(x - x0, p))  # Shape (Nx, Np)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 0.9)
ax.set_xlabel('z')
ax.set_ylabel(r'$|\psi(z,t)|^2$')
ax.grid(True)

line, = ax.plot([], [], lw=2, color='blue')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

prob_density = np.zeros((num_times, Nx))
for i, t in enumerate(t_vals):
    phase_p = np.exp(-1j * (p**2 / (2 * m)) * t / ħ)
    integrand = f_p * phase_p
    psi = prefactor * dp * (M @ integrand)
    prob_density[i] = np.abs(psi)**2

def update(frame):
    line.set_data(x, prob_density[frame])
    time_text.set_text(f'Time t = {t_vals[frame]:.2f}')
    return line, time_text

ani = FuncAnimation(fig, update, frames=num_times, 
                    interval=50, blit=True)
plt.tight_layout()
ani.save('prob_simulation.mp4', writer='ffmpeg', fps=30, dpi=300)
