import numpy as np
import os

# Constants
ħ = 1.0                   # Reduced Planck's constant
m_val = 1.0               # Mass
c = 1.0                   # Speed of light
p0 = 5.0                  # Central momentum
σ_p = 1.0                 # Momentum space width

# Calculate E0
E0 = np.sqrt(p0**2 * c**2 + m_val**2 * c**4)

# Time settings
t_min = 0.0
t_max = 60.0              # Time range (s)
num_times = 6            # Number of time frames
t_vals = np.linspace(t_min, t_max, num_times)

# Position grid
Nx = 4000                 # Number of position points
x_min = -5.0
x_max = 65.0
z_vals = np.linspace(x_min, x_max, Nx)


def calculate_rho_normalized(z, t):
    """
    Calculate ρ(z,t) using the complete normalized analytical expression
    """
    # Auxiliary terms
    A = (m_val**4 * c**12 * t**2) / (4 * ħ**2 * E0**6)
    B = 1 / (16 * σ_p**4)
    denom = A + B
    
    z_shifted = z - (p0 * c**2 * t) / E0
    
    # Exponential factor
    exp_arg = -(1 / (2 * ħ**2 * σ_p**2) * z_shifted**2) / (4 * denom)
    exp_factor = np.exp(exp_arg)
    
    # First bracket term
    term1_b1 = E0 / ħ
    term2_b1 = ((m_val**2 * c**6) / (4 * ħ**2 * E0**3) * (1 / (4 * σ_p))) / denom
    term3_b1_num = (1 / ħ**2) * z_shifted * (p0 * c**2 / E0) * (m_val**2 * c**6 * t) / (2 * ħ * E0**3)
    term3_b1 = term3_b1_num / (2 * denom)
    term4_b1_num = (2 * m_val**2 * c**6 / (ħ**3 * E0**3)) * z_shifted**2 * (B - (m_val**4 * c**12 * t**2) / (4 * ħ**2 * E0**6))
    term4_b1 = term4_b1_num / (16 * denom**2)
    
    bracket1 = term1_b1 - term2_b1 + term3_b1 - term4_b1
    
    # Second bracket term (normalization factor inverse)
    term1_b2 = E0 / ħ
    term2_b2 = ((m_val**2 * c**6) / (4 * ħ**2 * E0**3) * (1 / (4 * σ_p))) / denom
    term3_b2_num = (m_val**2 * c**6 / (ħ * E0**3)) * σ_p**2 * (B - (m_val**4 * c**12 * t**2) / (4 * ħ**2 * E0**6))
    term3_b2 = term3_b2_num / (2 * denom)
    
    bracket2 = term1_b2 - term2_b2 - term3_b2
    
    # Square root term (inverse)
    sqrt_term = np.sqrt(np.pi * 8 * ħ**2 * σ_p**2 * denom)
    
    # Complete expression
    rho = exp_factor * bracket1 / (bracket2 * sqrt_term)
    
    return rho
# Create directory for output
os.makedirs("rho_data", exist_ok=True)

# Calculate and export ρ(z,t) for each time
for i, t in enumerate(t_vals):
    print(f"Calculating ρ for t = {t:.3f}")
    
    # Calculate ρ for all z values at this time using the direct formula
    rho_vals = np.array([calculate_rho_normalized(z, t) for z in z_vals])
    
    # Format time label
    t_label = f"{int(t*1000):04d}"
    
    # Save data
    filename = f"rho_data/rho_t_{t_label}.dat"
    np.savetxt(filename, np.column_stack((z_vals, rho_vals)), 
               header='z rho_density', comments='', fmt='%.6e')
    
    print(f"Exported: {filename}")

print("\nCalculation complete!")
print(f"E0 = {E0:.6e} J")
print(f"Files saved in 'rho_data/' directory")
