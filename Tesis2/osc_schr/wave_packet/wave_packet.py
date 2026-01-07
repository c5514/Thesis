import numpy as np
import os

ħ = 1.0
m = 1.0
p0 = 5.0
x0 = 0.0
t_min = 0.0
t_max = 2.0
num_times = 10000
t_vals = np.linspace(t_min, t_max, num_times)
θ = 0.5
m1 = 1.0
m2 = 2.0
Δm_inv = (1/m1 - 1/m2)
sin2_2θ = (np.sin(2*θ))**2

P_ab = sin2_2θ * np.sin((p0**2 * t_vals * Δm_inv) / (2 * ħ))**2
P_aa = 1 - P_ab

osc_file = "oscillation_probabilities.dat"
np.savetxt(osc_file, np.column_stack((t_vals, P_ab, P_aa)),
           header='t P_alpha_to_beta P_alpha_to_alpha', 
           comments='', fmt=['%.6e', '%.6e', '%.6e'])
print(f"Exported: {osc_file}")
