import numpy as np
import os

ħ = 1.0
m = 1.0
p0 = 5.0
σ_p = 1.0
x0 = 0.0

t_min = 0.0
t_max = 2.0
num_times = 6
t_vals = np.linspace(t_min, t_max, num_times)

Nx = 4000
x_min = -2.0
x_max = 15.0
x = np.linspace(x_min, x_max, Nx)

Np = 8000
p_min = p0 - 10 * σ_p
p_max = p0 + 10 * σ_p
p = np.linspace(p_min, p_max, Np)
dp = p[1] - p[0]

f_p = (1.0 / ((2 * np.pi) ** 0.25 * np.sqrt(σ_p))) * np.exp(-(p - p0) ** 2 / (4 * σ_p ** 2))
prefactor = 1.0 / np.sqrt(2 * np.pi)
M = np.exp(1j * np.outer(x - x0, p))

psi = np.zeros((num_times, Nx), dtype=complex)
for i, t in enumerate(t_vals):
    phase_p = np.exp(-1j * (p ** 2 / (2 * m)) * t / ħ)
    integrand = f_p * phase_p
    psi[i, :] = prefactor * dp * (M @ integrand)

real_part = np.real(psi)
imag_part = np.imag(psi)
prob_density = np.abs(psi)**2

os.makedirs("wave_data", exist_ok=True)

for i, t in enumerate(t_vals):
    t_label = f"{int(t*1000):04d}"
    filename = f"wave_data/t_{t_label}.dat"
    np.savetxt(filename, np.column_stack((x, prob_density[i])), 
               header='x prob_density', comments='', fmt='%.6e')
    print(f"Exported: {filename}")

for i, t in enumerate(t_vals):
    t_label = f"{int(t*1000):04d}"
    filename = f"wave_data/t_im_re_{t_label}.dat"
    np.savetxt(filename, np.column_stack((x, real_part[i], imag_part[i])), 
               header='x real_part imag_part', comments='', fmt='%.6e')
    print(f"Exported: {filename}")
