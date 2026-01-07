import numpy as np
import os

# Constants
m_val = 1.0               # Mass
c = 1.0                   # Speed of light
p0 = 20.0                # Central momentum
σ_p = 1.0# Momentum space width

# Momentum grid
Np = 4000                 # Number of momentum points
p_min = 0       # Start at p0 - 5σ
p_max = 30        # End at p0 + 5σ
p_vals = np.linspace(p_min, p_max, Np)

def calculate_rho_p(p):
    """
    Calculate ρ_p(p) using the analytical expression
    """
    # Energy term
    E_p = np.sqrt(p**2 * c**2 + m_val**2 * c**4)
    
    # Prefactor
    prefactor = E_p / (m_val * c**2 * np.sqrt(σ_p * np.sqrt(2 * np.pi)))
    
    # Exponential term
    exp_arg = -(p - p0)**2 / (4 * σ_p**2)
    exp_factor = np.exp(exp_arg)
    
    return prefactor * exp_factor

# Create directory for output
os.makedirs("rho_p_data", exist_ok=True)

print(f"Calculating ρ_p(p) for {Np} momentum points")
print(f"Momentum range: [{p_min:.3f}, {p_max:.3f}]")

# Calculate ρ_p for all momentum values
rho_p_vals = np.array([calculate_rho_p(p) for p in p_vals])

# Save data
filename = "rho_p_data/rho_p.dat"
np.savetxt(filename, np.column_stack((p_vals, rho_p_vals)), 
           header='p rho_p', comments='', fmt='%.6e')

print(f"\nExported: {filename}")
print(f"p0 = {p0:.6f}")
print(f"σ_p = {σ_p:.6f}")
print(f"File saved in 'rho_p_data/' directory")
