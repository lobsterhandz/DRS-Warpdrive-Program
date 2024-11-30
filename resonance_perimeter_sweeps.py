import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal.windows import hamming
from scipy.fftpack import fft
import os
import concurrent.futures

# Define the parameter ranges for sweeps
lambda_res_range = np.linspace(1.0, 5.0, 5)  # Reduced Nonlinear coupling strength range for faster initial testing
gamma_range = np.linspace(0.1, 0.5, 5)       # Reduced Damping factor range for faster initial testing
higher_dim_range = np.linspace(0.5, 1.5, 3)   # Reduced Higher-dimensional influence range for faster initial testing

# Set up constants for oscillatory behavior
omega_1 = 2.0 * np.pi  # Angular frequency of primary oscillation
omega_2 = 4.0 * np.pi  # Angular frequency of secondary, higher frequency oscillation
k = 1.0  # Wavenumber for spatial variation
phi = np.pi / 3  # Phase constant
noise_factor_initial = 0.1  # Initial stochastic fluctuations

# Define the time frame and initial conditions
t_span = (0, 50)  # Time from 0 to 50 units
t_eval = np.linspace(0, 50, 200)  # Reduced Evaluation points in time for faster computation
initial_conditions = [1.0, 0.8]  # Initial conditions for A and B

# Directory to store data
output_dir = "parameter_sweep_data"
os.makedirs(output_dir, exist_ok=True)

# Define the differential equations for oscillatory, nonlinear, and higher-dimensional components
def dynamic_system(t, y, gamma, lambda_res, higher_dim_factor):
    A_t, B_t = y  # Unpacking current amplitudes
    
    # Time-dependent damping factor
    gamma_t = gamma * np.exp(-0.01 * t)
    
    # Time-dependent stochastic noise factor to simulate decreasing noise over time
    noise_factor = noise_factor_initial * np.exp(-0.01 * t)
    
    # Oscillatory contributions with damping
    dA_dt = -gamma_t * A_t * np.cos(omega_1 * t + k)  # Damped oscillatory effect on A
    dB_dt = -gamma_t * B_t * np.sin(omega_1 * t + k)  # Damped oscillatory effect on B
    
    # Nonlinear resonant term contributions (amplified)
    nonlinear_effect = lambda_res * np.cos(phi * omega_1 * t) * np.exp(-gamma_t * t)
    
    # Higher-dimensional contribution (modeled as an additional effect)
    higher_dim_effect = higher_dim_factor * np.cos(omega_2 * t + k * t) * np.exp(-gamma_t * t)
    
    # Coupling between frequency modes
    frequency_interaction = 0.5 * np.sin(omega_1 * t) * np.cos(omega_2 * t)
    
    # Stochastic noise addition
    stochastic_noise = noise_factor * np.random.normal(0, 1)
    
    # Calculating the changes over time for A_t and B_t
    dA_dt += nonlinear_effect + higher_dim_effect + frequency_interaction + stochastic_noise
    dB_dt += nonlinear_effect - higher_dim_effect + frequency_interaction + stochastic_noise
    
    return [dA_dt, dB_dt]

# Function to calculate Lyapunov exponent for chaos detection
def lyapunov_exponent(system, t_span, initial_conditions, params, epsilon=1e-9, delta_t=0.5):
    t0, tf = t_span
    t_vals = np.arange(t0, tf, delta_t)
    
    y0 = np.array(initial_conditions)
    perturbed_y0 = y0 + epsilon  # Slightly perturbed initial conditions
    
    lyapunov_sum = 0.0
    num_steps = len(t_vals)
    
    for i in range(num_steps - 1):
        sol1 = solve_ivp(system, (t_vals[i], t_vals[i+1]), y0, args=params, method='RK45')
        sol2 = solve_ivp(system, (t_vals[i], t_vals[i+1]), perturbed_y0, args=params, method='RK45')
        
        if sol1.success and sol2.success:
            dist = np.linalg.norm(sol2.y[:, -1] - sol1.y[:, -1])
            if dist == 0:
                return None  # Avoid log of zero
            y0 = sol1.y[:, -1]
            perturbed_y0 = sol2.y[:, -1] + epsilon * (perturbed_y0 - y0) / np.linalg.norm(perturbed_y0 - y0)  # Re-normalize perturbation
            lyapunov_sum += np.log(dist / epsilon)
        else:
            return None  # Integration failed
    
    lyapunov_exp = lyapunov_sum / (delta_t * num_steps)
    return lyapunov_exp

# Function to perform a single parameter sweep iteration
def perform_sweep_iteration(lambda_res, gamma, higher_dim_factor):
    params = (gamma, lambda_res, higher_dim_factor)
    
    # Solve the dynamic system
    solution = solve_ivp(dynamic_system, t_span, initial_conditions, t_eval=t_eval, args=params, method='RK45')
    
    if solution.success:
        # Calculate Lyapunov exponent
        lyapunov_exp = lyapunov_exponent(dynamic_system, t_span, initial_conditions, params)
        
        # Store results
        return {
            'lambda_res': lambda_res,
            'gamma': gamma,
            'higher_dim_factor': higher_dim_factor,
            'lyapunov_exp': lyapunov_exp,
            'A_final': solution.y[0, -1],
            'B_final': solution.y[1, -1]
        }
    return None

# Main function to perform parameter sweep
if __name__ == "__main__":
    # Perform parameter sweep in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for lambda_res in lambda_res_range:
            for gamma in gamma_range:
                for higher_dim_factor in higher_dim_range:
                    futures.append(executor.submit(perform_sweep_iteration, lambda_res, gamma, higher_dim_factor))
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'parameter_sweep_results.csv'), index=False)

    print("Parameter sweep completed. Data saved to 'parameter_sweep_results.csv'")
