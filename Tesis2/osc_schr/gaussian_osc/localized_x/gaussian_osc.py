import numpy as np
import os
ħ = 1.0
m = 1.0
p0 = 5.0
sg = 0.7
t_min = 0.0
t_max = 4.0
num_times = 10000
t_vals = np.linspace(t_min, t_max, num_times)
θ = 0.5
m1 = 1.0
m2 = 2.0
Δm_inv = (1/m1 - 1/m2)
sin2theta_cos2theta = 2 * np.sin(θ)**2 * np.cos(θ)**2
def calculate_P_ab(t):
    σ = sg
    σ2 = σ**2
    σ4 = σ**4
    term_A = 1/(4*σ4)
    term_B = (t**2/(4*ħ**2)) * Δm_inv**2
    denom = term_A + term_B
    prefactor = 1/(σ * np.sqrt(2 * np.sqrt(denom)))
    exp_numerator = (p0**2 * t**2)/(8*ħ**2*σ2) * Δm_inv**2
    exp_factor = np.exp(-exp_numerator / denom)
    arctan_term = 0.5 * np.arctan((t*σ2/ħ) * Δm_inv)
    cosine_term = (p0**2 * t)/(8*ħ*σ4) * (1/denom) * Δm_inv
    cos_arg = arctan_term + cosine_term
    P_ab = sin2theta_cos2theta * (1 - prefactor * exp_factor * np.cos(cos_arg))
    return P_ab

P_ab_vals = np.array([calculate_P_ab(t) for t in t_vals])
P_aa_vals = 1 - P_ab_vals
sin2theta_cos2theta_vals = np.full_like(t_vals, sin2theta_cos2theta)

osc_file = "oscillation_probabilities_gaussian.dat"
np.savetxt(osc_file, np.column_stack((t_vals, P_ab_vals, P_aa_vals, sin2theta_cos2theta_vals)),
           header='t P_alpha_to_beta P_alpha_to_alpha 2sin2theta_cos2theta', 
           comments='', fmt=['%.6e', '%.6e', '%.6e', '%.6e'])
print(f"Exported: {osc_file}")
print(f"2sin²θcos²θ = {sin2theta_cos2theta:.6f}")
