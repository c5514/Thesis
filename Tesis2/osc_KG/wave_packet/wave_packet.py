import numpy as np
import os

ħ = 1.0
m = 1.0
c = 1.0
p = 5.0
t_min = 0.0
t_max = 60.0
num_times = 10000
t_vals = np.linspace(t_min, t_max, num_times)
θ = 0.5
m1 = 1.0
m2 = 2.0

P_ab = (np.sin(2*θ))**2 * np.sin(((m1**2 - m2**2) * c**3 * t_vals) / (4 * p * ħ))**2
P_aa = 1 - P_ab

osc_file = "oscillation_probabilities.dat"
np.savetxt(osc_file, np.column_stack((t_vals, P_ab, P_aa)),
           header='t P_alpha_to_beta P_alpha_to_alpha', 
           comments='', fmt=['%.6e', '%.6e', '%.6e'])
print(f"Exported: {osc_file}")
