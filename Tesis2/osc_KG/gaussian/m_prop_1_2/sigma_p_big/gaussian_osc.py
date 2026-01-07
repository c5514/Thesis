import numpy as np
import os

ħ = 1.0
c = 1.0
p0 = 5.0
sg = 0.7
t_min = 0.0
t_max = 60.0
num_times = 10000
t_vals = np.linspace(t_min, t_max, num_times)
θ = 0.5
m1 = 1.0
m2 = 2.0
E_01 = np.sqrt(p0**2 * c**2 + m1**2 * c**4)
E_02 = np.sqrt(p0**2 * c**2 + m2**2 * c**4)

def calculate_real_integral(t):
    σ_p = sg
    σ_p2 = σ_p**2
    σ_p4 = σ_p**4
    term1 = (m1**2 * c**6) / (2 * E_01**3)
    term2 = (m2**2 * c**6) / (2 * E_02**3)
    term_diff = term1 - term2
    denom_base = 1/σ_p4 - (t**2/ħ**2) * term_diff**2
    prefactor = 1 / (np.sqrt(np.sqrt(denom_base)) * σ_p)
    exp_numerator = (t**2 * p0**2 * c**4 / ħ**2) * (1/E_01 - 1/E_02)**2
    exp_denom = 4 * σ_p2 * denom_base
    exp_factor = np.exp(-exp_numerator / exp_denom)
    cos_term1 = t/ħ * (E_01 - E_02)
    cos_term2_numerator = term_diff
    cos_term2_denom = 4 * σ_p2 * denom_base
    cos_term2 = (t/ħ) * (cos_term2_numerator / cos_term2_denom)
    arctan_term = 0.5 * np.arctan((t * σ_p2 / ħ) * term_diff)
    cos_arg = cos_term1 + cos_term2 - arctan_term
    real_integral = prefactor * exp_factor * np.cos(cos_arg)
    
    return real_integral

def calculate_P_ab(t):
    real_int = calculate_real_integral(t)
    P_ab = 2 * np.sin(θ)**2 * np.cos(θ)**2 * (1 - real_int)
    return P_ab

P_ab_vals = np.array([calculate_P_ab(t) for t in t_vals])
P_aa_vals = 1 - P_ab_vals

print(f"P_ab en t=0: {calculate_P_ab(0)}")
print(f"P_ab en t=0 debería ser: 0")

osc_file = "prob_KG.dat"
np.savetxt(osc_file, np.column_stack((t_vals, P_ab_vals, P_aa_vals)),
           header='t P_alpha_to_beta P_alpha_to_alpha', 
           comments='', fmt=['%.6e', '%.6e', '%.6e'])
print(f"Exported: {osc_file}")
