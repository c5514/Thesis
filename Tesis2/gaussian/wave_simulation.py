import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

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

Np = 1000
p_min = p0 - 10 * σ_p
p_max = p0 + 10 * σ_p
p = np.linspace(p_min, p_max, Np)
dp = p[1] - p[0]

f_p = (1.0 / ((2 * np.pi)**0.25 * np.sqrt(σ_p))) * np.exp(-(p - p0)**2 / (4 * σ_p**2))

prefactor = 1.0 / np.sqrt(2 * np.pi)
M = np.exp(1j * np.outer(x - x0, p))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.0, 1.0)    # Real component range
ax.set_zlim(-1.0, 1.0)    # Imaginary component range

ax.set_xlabel('z', labelpad=15)
ax.set_ylabel('Re(ψ)', labelpad=15)
ax.set_zlabel('Im(ψ)', labelpad=15)

real_parts = np.zeros((num_times, Nx))
imag_parts = np.zeros((num_times, Nx))

for i, t in enumerate(t_vals):
    phase_p = np.exp(-1j * (p**2 / (2 * m)) * t / ħ)
    integrand = f_p * phase_p
    psi = prefactor * dp * (M @ integrand)
    real_parts[i] = np.real(psi)
    imag_parts[i] = np.imag(psi)

real_line, = ax.plot(x, real_parts[0], np.zeros(Nx), 'b-', lw=1.5)
imag_line, = ax.plot(x, np.zeros(Nx), imag_parts[0], 'r-', lw=1.5)
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

ax.plot([x_min, x_max], [0, 0], [0, 0], 'k-', alpha=0.3)  # x-axis
ax.plot([0, 0], [-0.5, 0.5], [0, 0], 'k-', alpha=0.3)     # y-axis (real)
ax.plot([0, 0], [0, 0], [-0.5, 0.5], 'k-', alpha=0.3)     # z-axis (imag)

def update(frame):
    real_line.set_data(x, real_parts[frame])
    real_line.set_3d_properties(np.zeros(Nx))
    imag_line.set_data(x, np.zeros(Nx))
    imag_line.set_3d_properties(imag_parts[frame])
    time_text.set_text(f't = {t_vals[frame]:.2f}')
    
    ax.view_init(elev=30, azim=250)
    
    return real_line, imag_line, time_text

ani = FuncAnimation(fig, update, frames=num_times, 
                    interval=50, blit=True)

plt.tight_layout()
ani.save('wave_simulation.mp4', writer='ffmpeg', fps=30, dpi=200)
