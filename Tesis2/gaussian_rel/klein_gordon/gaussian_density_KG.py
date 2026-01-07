import numpy as np
import os

# Constants
ħ = 1.0                   # Reduced Planck's constant
m_val = 1.0               # Mass (renamed to avoid conflict)
c = 1.0                   # Speed of light
p0 = 5.0                  # Central momentum
σ_p = 1.0                 # Momentum space width

# Calculate E0
E0 = np.sqrt(p0**2 * c**2 + m_val**2 * c**4)

# Time settings
t_min = 0.0
t_max = 2.0               # Time range
num_times = 6             # Number of time frames
t_vals = np.linspace(t_min, t_max, num_times)

# Position grid
Nx = 4000                 # Number of position points
x_min = -5.0
x_max = 15.0
z_vals = np.linspace(x_min, x_max, Nx)

# Precompute constants
prefactor_const = 1.0 / np.sqrt(σ_p * 2 * np.pi * ħ * np.sqrt(2 * np.pi))
m2c6 = m_val**2 * c**6
E0_3 = E0**3

def calculate_phi(z, t):
    """Calculate the wave function ϕ(z,t) using the provided expression"""
    if t == 0:
        # Handle t=0 case separately
        phase = np.exp(-1j/ħ * (-p0 * z))
        sqrt_term = np.sqrt(np.pi / (1/(4*σ_p**2)))
        exp_term = np.exp(- (z**2 / ħ**2) / (4/(4*σ_p**2)))
        return phase * prefactor_const * sqrt_term * exp_term
    
    # Complex term in denominator
    complex_term = 1j * m2c6 * t / (ħ * 2 * E0_3) + 1/(4*σ_p**2)
    
    # Phase factor
    phase = np.exp(-1j/ħ * (E0*t - p0*z))
    
    # Square root term
    sqrt_term = np.sqrt(np.pi / complex_term)
    
    # Exponential term
    z_term = z - (p0 * c**2 * t) / E0
    exp_arg = - (z_term**2 / ħ**2) / (4 * complex_term)
    exp_term = np.exp(exp_arg)
    
    return phase * prefactor_const * sqrt_term * exp_term

def calculate_dphi_dt(z, t):
    """Calculate ∂ϕ/∂t using the provided expression"""
    if t == 0:
        # Use small t approximation for t=0
        t_small = 1e-10
        return calculate_dphi_dt(z, t_small)
    
    # Common complex term
    complex_term = 1j * m2c6 * t / (ħ * 2 * E0_3) + 1/(4*σ_p**2)
    z_term = z - (p0 * c**2 * t) / E0
    
    # Prefactor for all terms
    base_prefactor = np.exp(-1j/ħ * (E0*t - p0*z)) / np.sqrt(σ_p * 2 * np.pi * ħ * np.sqrt(2 * np.pi))
    
    # Term 1: -i/ħ E0 term
    sqrt_term1 = np.sqrt(np.pi / complex_term)
    exp_arg1 = - (z_term**2 / ħ**2) / (4 * complex_term)
    term1 = (-1j/ħ * E0) * base_prefactor * sqrt_term1 * np.exp(exp_arg1)
    
    # Term 2: i m²c⁶/(4ħE0³) term
    sqrt_term2 = np.sqrt(np.pi / complex_term**3)
    exp_arg2 = exp_arg1  # Same exponential as term1
    term2 = (1j * m2c6 / (4 * ħ * E0_3)) * base_prefactor * sqrt_term2 * np.exp(exp_arg2)
    
    # Term 3: Complex derivative term
    sqrt_term3 = np.sqrt(np.pi / complex_term)
    exp_arg3 = exp_arg1  # Same exponential as term1
    
    # Numerator for term3
    numerator = (-8/ħ**2 * z_term * (p0*c**2/E0) * complex_term - 
                 1j * 2 * m2c6 / (ħ**3 * E0_3) * z_term**2)
    denominator = 16 * complex_term**2
    
    term3 = base_prefactor * sqrt_term3 * np.exp(exp_arg3) * numerator / denominator
    
    return term1 + term2 + term3

def calculate_rho(z, t):
    """Calculate the probability density ρ(z,t)"""
    phi = calculate_phi(z, t)
    dphi_dt = calculate_dphi_dt(z, t)
    
    # Complex conjugate
    phi_star = np.conj(phi)
    dphi_dt_star = np.conj(dphi_dt)
    
    # Calculate ρ
    rho = (1j * ħ / (2 * m_val * c**2)) * (phi_star * dphi_dt - phi * dphi_dt_star)
    
    return np.real(rho)  # ρ should be real

# Create directory for output
os.makedirs("rho_data", exist_ok=True)

# Calculate and export ρ(z,t) for each time
for i, t in enumerate(t_vals):
    print(f"Calculating ρ for t = {t:.3f}")
    
    # Calculate ρ for all z values at this time
    rho_vals = np.array([calculate_rho(z, t) for z in z_vals])
    
    # Format time label
    t_label = f"{int(t*1000):04d}"
    
    # Save data
    filename = f"rho_data/rho_t_{t_label}.dat"
    np.savetxt(filename, np.column_stack((z_vals, rho_vals)), 
               header='z rho_density', comments='', fmt='%.6e')
    
    print(f"Exported: {filename}")

print("\nCalculation complete!")
print(f"E0 = {E0:.6f}")
print(f"Files saved in 'rho_data/' directory")
