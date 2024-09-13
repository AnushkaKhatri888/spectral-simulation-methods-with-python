import numpy as np
from muFFT import FFT
import matplotlib.pyplot as plt

# Simulation parameters
grid_points = 32  # Grid size (32x32x32)
domain_size = 1.0  # Box length
viscosity = 1/1600  # Viscosity
time_step = 0.01  # Time step
t_final = 5.0  # Total simulation time
velocity_amplitude = 1.0  # Amplitude for initial velocity field

# Set up FFT
nb_grid_pts = (grid_points, grid_points, grid_points)
fft_transform = FFT(nb_grid_pts, engine='pocketfft')

# Grid spacing
grid_spacing = domain_size / grid_points

# Compute wavevectors
wavevector = (2 * np.pi * fft_transform.fftfreq.T / grid_spacing).T
zero_wavevector = (wavevector.T == np.zeros(3, dtype=int)).T.all(axis=0)
wavevector_sq = np.sum(wavevector ** 2, axis=0)


def create_initial_velocity_field():
    """
    Create the initial velocity field in Fourier space using random values
    and ensure the field is incompressible.
    
    Returns:
    np.ndarray: Initial velocity field in Fourier space.
    """
    random_field = np.zeros((3,) + fft_transform.nb_fourier_grid_pts, dtype=complex)
    rng = np.random.default_rng()
    random_field.real = rng.standard_normal(random_field.shape)
    random_field.imag = rng.standard_normal(random_field.shape)
    
    fac = np.zeros_like(wavevector_sq)
    fac[np.logical_not(zero_wavevector)] = velocity_amplitude * \
        wavevector_sq[np.logical_not(zero_wavevector)] ** (-5 / 6)
    random_field *= fac
    
    k_dot_u = np.sum(wavevector * random_field, axis=0)
    for i in range(3):
        random_field[i] -= (wavevector[i] * k_dot_u) / np.where(wavevector_sq == 0, 1, wavevector_sq)
    
    random_field[:, zero_wavevector] = 0
    
    return random_field

def navier_stokes_rhs(u_hat):
    """
    Compute the right-hand side (RHS) of the Navier-Stokes equations in Fourier space.
    
    Parameters:
    u_hat (np.ndarray): Current velocity field in Fourier space.
    
    Returns:
    np.ndarray: Right-hand side of the Navier-Stokes equations.
    """
    u = np.array([fft_transform.ifft(u_hat[i]) for i in range(3)])
    
    nonlinear = np.array([
        fft_transform.fft(u[1]*u[2]),
        fft_transform.fft(u[2]*u[0]),
        fft_transform.fft(u[0]*u[1])
    ])
    
    rhs = -1j * np.cross(wavevector, nonlinear, axis=0)
    rhs -= viscosity * wavevector_sq * u_hat
    
    k_dot_rhs = np.sum(wavevector * rhs, axis=0)
    for i in range(3):
        rhs[i] -= (wavevector[i] * k_dot_rhs) / np.where(wavevector_sq == 0, 1, wavevector_sq)
    
    return rhs


def rk4_step(u_hat, time_step):
    """
    Perform a single Runge-Kutta 4th order (RK4) time step.
    
    Parameters:
    u_hat (np.ndarray): Current velocity field in Fourier space.
    time_step (float): Size of the time step.
    
    Returns:
    np.ndarray: Updated velocity field after one RK4 step.
    """
    k1 = navier_stokes_rhs(u_hat)
    k2 = navier_stokes_rhs(u_hat + 0.5 * time_step * k1)
    k3 = navier_stokes_rhs(u_hat + 0.5 * time_step * k2)
    k4 = navier_stokes_rhs(u_hat + time_step * k3)
    return u_hat + (time_step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def apply_forcing(u_hat):
    """
    Apply forcing to the velocity field to maintain turbulence.
    
    Parameters:
    u_hat (np.ndarray): Current velocity field in Fourier space.
    
    Returns:
    np.ndarray: Velocity field after forcing is applied.
    """
    mask = (np.sqrt(wavevector_sq) <= 2 * np.pi / domain_size)
    u_hat[:, mask] *= np.exp(viscosity * wavevector_sq[mask] * time_step)
    return u_hat


def normalize_velocity(u_hat):
    """
    Normalize the velocity field to maintain constant total energy.
    
    Parameters:
    u_hat (np.ndarray): Current velocity field in Fourier space.
    
    Returns:
    np.ndarray: Normalized velocity field.
    """
    energy = np.sum(np.abs(u_hat) ** 2)
    return u_hat * np.sqrt(1 / energy)


def compute_spectra(u_hat):
    """
    Compute the energy and dissipation spectra.
    
    Parameters:
    u_hat (np.ndarray): Velocity field in Fourier space.
    
    Returns:
    tuple: Energy spectrum and dissipation spectrum.
    """
    energy_spectrum = np.zeros(grid_points // 2)
    dissipation_spectrum = np.zeros(grid_points // 2)
    
    for i in range(grid_points // 2):
        shell = (i <= np.sqrt(wavevector_sq)) & (np.sqrt(wavevector_sq) < i + 1)
        energy_spectrum[i] = 0.5 * np.sum(np.abs(u_hat[:, shell]) ** 2) / (grid_points ** 3)
        dissipation_spectrum[i] = 2 * viscosity * np.sum(wavevector_sq[shell] * np.abs(u_hat[:, shell]) ** 2) / (grid_points ** 3)
    
    return energy_spectrum, dissipation_spectrum

# Function to save the plots with better names and visuals
def save_plots(t, u_hat):
    """
    Save plots of the velocity field and energy/dissipation spectra.
    
    Parameters:
    t (float): Current time in the simulation.
    u_hat (np.ndarray): Current velocity field in Fourier space.
    """
    u = np.array([fft_transform.ifft(u_hat[i]) for i in range(3)])
    
    # Visualize velocity field with enhanced visuals
    plt.figure(figsize=(10,10))
    plt.imshow(np.sqrt(np.sum(u[:,:,:,grid_points//2]**2, axis=0)), cmap='plasma')  # Colormap for better visualization
    plt.colorbar()
    plt.title(f'Velocity Magnitude at t={t:.2f}', fontsize=14, color='darkblue')
    velocity_field_filename = f'velocity_field_at_time_{t:.2f}.png'
    plt.savefig(velocity_field_filename)
    plt.close()

    # Compute and plot spectra with better layout
    energy_spectrum, dissipation_spectrum = compute_spectra(u_hat)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Energy spectrum plot
    ax1.loglog(range(1, grid_points // 2 + 1), energy_spectrum, label='Energy Spectrum', color='blue', marker='o', linestyle='--', linewidth=2)
    ax1.set_xlabel('Wavenumber k', fontsize=12)
    ax1.set_ylabel('Energy Spectrum', fontsize=12)
    ax1.legend()
    
    # Dissipation spectrum plot
    ax2.loglog(range(1, grid_points // 2 + 1), dissipation_spectrum, label='Dissipation Spectrum', color='red', marker='x', linestyle='-', linewidth=2)
    ax2.set_xlabel('Wavenumber k', fontsize=12)
    ax2.set_ylabel('Dissipation Spectrum', fontsize=12)
    ax2.legend()

    # Adjust layout and space between subplots
    plt.subplots_adjust(wspace=0.4)  # Increased spacing between subplots
    fig.suptitle(f'Energy and Dissipation Spectra at t={t:.2f}', fontsize=16, color='darkblue')
    
    spectra_filename = f'energy_and_dissipation_spectra_at_time_{t:.2f}.png'
    plt.savefig(spectra_filename)
    plt.close()

    print(f'Saved plots: {velocity_field_filename}, {spectra_filename}')

# Initialize velocity field
u_hat = create_initial_velocity_field()

# Main simulation loop
t = 0
plot_times = [0, t_final / 4, t_final / 2, 3 * t_final / 4, t_final]
plot_index = 0

# Save initial state
save_plots(t, u_hat)
plot_index += 1

while t < t_final:
    u_hat = rk4_step(u_hat, time_step)
    u_hat = apply_forcing(u_hat)
    u_hat = normalize_velocity(u_hat)
    t += time_step
    
    if plot_index < len(plot_times) and t >= plot_times[plot_index]:
        save_plots(t, u_hat)
        plot_index += 1
    
    print(f"Time: {t:.2f} / {t_final}")

# Ensure we save the final state
if plot_index < len(plot_times):
    save_plots(t, u_hat)

print("Simulation complete.")
