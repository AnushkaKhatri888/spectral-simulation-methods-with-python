import numpy as np
from muFFT import FFT  # Import the FFT class from the muFFT library
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Simulation Parameters
grid_points = 32  # Number of grid points (32x32x32 grid)
domain_size = 1.0  # Physical size of the domain
viscosity = 1 / 1600  # Kinematic viscosity of the fluid
time_step = 0.01  # Time step for the simulation
t_final = 5.0  # Total simulation time
velocity_amplitude = 1.0  # Amplitude for the initial velocity field

# Set up FFT for the grid
nb_grid_pts = (grid_points, grid_points, grid_points)
fft_transform = FFT(nb_grid_pts, engine='pocketfft')  # Use 'pocketfft' engine for FFT computations

# Grid spacing
grid_spacing = domain_size / grid_points

# Compute wavevectors for Fourier transforms
wavevector = (2 * np.pi * fft_transform.fftfreq.T / grid_spacing).T
zero_wavevector = (wavevector.T == np.zeros(3, dtype=int)).T.all(axis=0)
wavevector_sq = np.sum(wavevector ** 2, axis=0)

# Initialize Velocity Field Function
def create_initial_velocity_field():
    """
    Create an initial random velocity field in Fourier space and ensure incompressibility.

    Returns:
    np.ndarray: The velocity field in Fourier space.
    """
    random_field = np.zeros((3,) + fft_transform.nb_fourier_grid_pts, dtype=complex)
    rng = np.random.default_rng()
    
    # Assign random values to real and imaginary parts
    random_field.real = rng.standard_normal(random_field.shape)
    random_field.imag = rng.standard_normal(random_field.shape)
    
    # Scale by the velocity amplitude and wavevector magnitude
    fac = np.zeros_like(wavevector_sq)
    fac[np.logical_not(zero_wavevector)] = velocity_amplitude * wavevector_sq[np.logical_not(zero_wavevector)] ** (-5 / 6)
    random_field *= fac
    
    # Project the velocity field to ensure incompressibility
    k_dot_u = np.sum(wavevector * random_field, axis=0)
    for i in range(3):
        random_field[i] -= (wavevector[i] * k_dot_u) / np.where(wavevector_sq == 0, 1, wavevector_sq)

    random_field[:, zero_wavevector] = 0  # Set zero-wavevector components to zero
    
    return random_field

# Aliasing Correction (2/3 Rule)
def apply_2thirds_rule(data):
    """
    Apply the 2/3 rule for aliasing correction by filtering out high wavevectors.
    
    Parameters:
    data (np.ndarray): Fourier data to be corrected.
    
    Returns:
    np.ndarray: Corrected data with high wavevectors filtered.
    """
    filter_mask = np.ones(data.shape, dtype=bool)
    for i in range(3):
        filter_mask[:, fft_transform.nb_fourier_grid_pts[i] // 3:2 * fft_transform.nb_fourier_grid_pts[i] // 3] = False
    return data * filter_mask

# Navier-Stokes Right-Hand-Side (RHS) Computation
def navier_stokes_rhs(u_hat, aliasing_correction=False):
    """
    Compute the right-hand side of the Navier-Stokes equations in Fourier space.
    
    Parameters:
    u_hat (np.ndarray): Velocity field in Fourier space.
    aliasing_correction (bool): Whether to apply aliasing correction using the 2/3 rule.
    
    Returns:
    np.ndarray: The right-hand side of the Navier-Stokes equations.
    """
    # Convert velocity field from Fourier space to real space
    u = np.array([fft_transform.ifft(u_hat[i]) for i in range(3)])
    
    if aliasing_correction:
        u = apply_2thirds_rule(u)  # Apply aliasing correction if required

    # Compute nonlinear term in real space and transform back to Fourier space
    nonlinear = np.array([
        fft_transform.fft(u[1] * u[2]),
        fft_transform.fft(u[2] * u[0]),
        fft_transform.fft(u[0] * u[1])
    ])
    
    if aliasing_correction:
        nonlinear = apply_2thirds_rule(nonlinear)  # Apply aliasing correction to the nonlinear term

    # Compute the right-hand side (RHS) in Fourier space
    rhs = -1j * np.cross(wavevector, nonlinear, axis=0)  # Compute the cross product in Fourier space
    rhs -= viscosity * wavevector_sq * u_hat  # Apply viscous damping
    
    # Project to ensure incompressibility (remove divergence)
    k_dot_rhs = np.sum(wavevector * rhs, axis=0)
    for i in range(3):
        rhs[i] -= (wavevector[i] * k_dot_rhs) / np.where(wavevector_sq == 0, 1, wavevector_sq)
    
    return rhs

# Fourth-order Runge-Kutta Scheme (RK4)
def rk4_step(u_hat, time_step, aliasing_correction=False):
    """
    Perform a single Runge-Kutta 4th order (RK4) time step.
    
    Parameters:
    u_hat (np.ndarray): Current velocity field in Fourier space.
    time_step (float): Size of the time step.
    aliasing_correction (bool): Whether to apply aliasing correction.
    
    Returns:
    np.ndarray: Updated velocity field after one RK4 step.
    """
    k1 = navier_stokes_rhs(u_hat, aliasing_correction)
    k2 = navier_stokes_rhs(u_hat + 0.5 * time_step * k1, aliasing_correction)
    k3 = navier_stokes_rhs(u_hat + 0.5 * time_step * k2, aliasing_correction)
    k4 = navier_stokes_rhs(u_hat + time_step * k3, aliasing_correction)
    return u_hat + (time_step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Forcing Function for Low-Wavenumber Modes
def apply_forcing(u_hat):
    """
    Apply forcing to the velocity field to maintain turbulence at low wavenumbers.
    
    Parameters:
    u_hat (np.ndarray): Current velocity field in Fourier space.
    
    Returns:
    np.ndarray: Updated velocity field after forcing is applied.
    """
    mask = (np.sqrt(wavevector_sq) <= 2 * np.pi / domain_size)  # Apply forcing at large scales
    u_hat[:, mask] *= np.exp(viscosity * wavevector_sq[mask] * time_step)
    return u_hat

# Normalize Velocity Field to Maintain Energy
def normalize_velocity(u_hat):
    """
    Normalize the velocity field to maintain constant total energy.
    
    Parameters:
    u_hat (np.ndarray): Current velocity field in Fourier space.
    
    Returns:
    np.ndarray: Normalized velocity field.
    """
    energy = np.sum(np.abs(u_hat) ** 2)  # Calculate total energy in Fourier space
    return u_hat * np.sqrt(1 / energy)  # Normalize the velocity field

# Spectrum Calculation for Energy and Dissipation
def compute_spectra(u_hat):
    """
    Compute the energy and dissipation spectra of the velocity field.
    
    Parameters:
    u_hat (np.ndarray): Velocity field in Fourier space.
    
    Returns:
    tuple: Energy spectrum and dissipation spectrum.
    """
    energy_spectrum = np.zeros(grid_points // 2)
    dissipation_spectrum = np.zeros(grid_points // 2)
    
    # Loop over wavenumbers to compute energy and dissipation spectra
    for i in range(grid_points // 2):
        shell = (i <= np.sqrt(wavevector_sq)) & (np.sqrt(wavevector_sq) < i + 1)
        energy_spectrum[i] = 0.5 * np.sum(np.abs(u_hat[:, shell]) ** 2) / (grid_points ** 3)
        dissipation_spectrum[i] = 2 * viscosity * np.sum(wavevector_sq[shell] * np.abs(u_hat[:, shell]) ** 2) / (grid_points ** 3)
    
    return energy_spectrum, dissipation_spectrum

# Function to save plots of the results
def save_plots(t, u_hat_no_correction, u_hat_with_correction):
    """
    Save plots of the velocity field and energy/dissipation spectra at time t.

    Parameters:
    t (float): Current simulation time.
    u_hat_no_correction (np.ndarray): Velocity field without aliasing correction.
    u_hat_with_correction (np.ndarray): Velocity field with aliasing correction.
    """
    # Compute energy and dissipation spectra for both cases
    energy_spectrum_no_correction, dissipation_spectrum_no_correction = compute_spectra(u_hat_no_correction)
    energy_spectrum_with_correction, dissipation_spectrum_with_correction = compute_spectra(u_hat_with_correction)

    # Plot the energy and dissipation spectra
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Energy Spectrum Plot
    ax1.loglog(range(1, grid_points // 2 + 1), energy_spectrum_no_correction, label='Energy (Without Correction)', color='deeppink', marker='o', linestyle='--', linewidth=2, markersize=8)
    ax1.loglog(range(1, grid_points // 2 + 1), energy_spectrum_with_correction, label='Energy (With Correction)', color='blue', marker='s', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel('Wavenumber (k)', fontsize=14)
    ax1.set_ylabel('Energy Spectrum', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.set_title('Energy Spectrum', fontsize=16)

    # Dissipation Spectrum Plot
    ax2.loglog(range(1, grid_points // 2 + 1), dissipation_spectrum_no_correction, label='Dissipation (Without Correction)', color='green', marker='x', linestyle='-', linewidth=2, markersize=8)
    ax2.loglog(range(1, grid_points // 2 + 1), dissipation_spectrum_with_correction, label='Dissipation (With Correction)', color='red', marker='D', linestyle='-', linewidth=2, markersize=8)
    ax2.set_xlabel('Wavenumber (k)', fontsize=14)
    ax2.set_ylabel('Dissipation Spectrum', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.set_title('Dissipation Spectrum', fontsize=16)

    fig.suptitle(f'Spectra Analysis at t={t:.2f}', fontsize=18, color='darkblue')
    plt.savefig(f'spectra_output_t{t:.2f}.png')
    plt.close()

    # Plot the velocity magnitude without aliasing correction
    u_real_no_correction = np.array([fft_transform.ifft(u_hat_no_correction[i]) for i in range(3)])
    plt.figure(figsize=(10, 10))
    plt.imshow(np.sqrt(np.sum(u_real_no_correction[:, :, :, grid_points // 2] ** 2, axis=0)), cmap='Blues')
    plt.colorbar(label='Velocity Magnitude')
    plt.title(f'Velocity Magnitude (No Correction) at t={t:.2f}', fontsize=14, color='darkblue')
    plt.savefig(f'velocity_no_correction_output_t{t:.2f}.png')
    plt.close()

    # Plot the velocity magnitude with aliasing correction
    u_real_with_correction = np.array([fft_transform.ifft(u_hat_with_correction[i]) for i in range(3)])
    plt.figure(figsize=(10, 10))
    plt.imshow(np.sqrt(np.sum(u_real_with_correction[:, :, :, grid_points // 2] ** 2, axis=0)), cmap='Oranges')
    plt.colorbar(label='Velocity Magnitude')
    plt.title(f'Velocity Magnitude (With Correction) at t={t:.2f}', fontsize=14, color='darkred')
    plt.savefig(f'velocity_with_correction_output_t{t:.2f}.png')
    plt.close()

    print(f'Saved plots for time t={t:.2f}')

# Initialize velocity fields
u_hat_no_correction = create_initial_velocity_field()
u_hat_with_correction = u_hat_no_correction.copy()

# Main simulation loop
t = 0
plot_times = [0, t_final / 4, t_final / 2, 3 * t_final / 4, t_final]
plot_index = 0

# Save initial plots
save_plots(t, u_hat_no_correction, u_hat_with_correction)
plot_index += 1

# Time-stepping loop
while t < t_final:
    u_hat_no_correction = rk4_step(u_hat_no_correction, time_step, aliasing_correction=False)
    u_hat_with_correction = rk4_step(u_hat_with_correction, time_step, aliasing_correction=True)
    u_hat_no_correction = apply_forcing(u_hat_no_correction)
    u_hat_with_correction = apply_forcing(u_hat_with_correction)
    u_hat_no_correction = normalize_velocity(u_hat_no_correction)
    u_hat_with_correction = normalize_velocity(u_hat_with_correction)
    t += time_step

    if plot_index < len(plot_times) and t >= plot_times[plot_index]:
        save_plots(t, u_hat_no_correction, u_hat_with_correction)
        plot_index += 1

# Ensure final state is saved
if plot_index < len(plot_times):
    save_plots(t, u_hat_no_correction, u_hat_with_correction)

print("Simulation complete.")
