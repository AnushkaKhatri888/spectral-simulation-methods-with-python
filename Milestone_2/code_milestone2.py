import numpy as np
import muFFT as mfft
import matplotlib.pyplot as plt

# Set up grid and FFT parameters
grid_points = 64  # Number of grid points in each direction
domain_size = 2 * np.pi  # Size of the simulation domain
viscosity = 0.01  # Kinematic viscosity of the fluid

fft_transform = mfft.FFT([grid_points, grid_points, grid_points])  # Initialize FFT object

# Generate wavenumbers
wave_numbers = fft_transform.fftfreq  # Get the wave numbers for FFT
kx, ky, kz = wave_numbers[0, :, :, :], wave_numbers[1, :, :, :], wave_numbers[2, :, :, :]  # Extract kx, ky, kz components
k_squared = kx**2 + ky**2 + kz**2  # Compute squared magnitude of wave numbers
k_squared[0, 0, 0] = 1  # Avoid division by zero by setting the (0,0,0) element

def compute_rhs(current_time, velocity_hat):
    """
    Compute the right-hand side of the Navier-Stokes equations in Fourier space.
    
    Parameters:
    current_time (float): Current time in the simulation.
    velocity_hat (ndarray): Velocity field in Fourier space.

    Returns:
    ndarray: The computed right-hand side in Fourier space.
    """
    velocity = fft_transform.ifft(velocity_hat)  # Transform velocity to real space

    # Compute nonlinear term in real space
    nonlinear_term = np.array([
        velocity[0] * np.gradient(velocity[0], axis=0) + velocity[1] * np.gradient(velocity[0], axis=1) + velocity[2] * np.gradient(velocity[0], axis=2),
        velocity[0] * np.gradient(velocity[1], axis=0) + velocity[1] * np.gradient(velocity[1], axis=1) + velocity[2] * np.gradient(velocity[1], axis=2),
        velocity[0] * np.gradient(velocity[2], axis=0) + velocity[1] * np.gradient(velocity[2], axis=1) + velocity[2] * np.gradient(velocity[2], axis=2)
    ])

    # Transform nonlinear term to Fourier space
    nonlinear_hat = fft_transform.fft(nonlinear_term)

    # Compute projection to ensure incompressibility
    projection_k = 1 - (kx**2 + ky**2 + kz**2) / k_squared

    # Compute right-hand side in Fourier space
    rhs = -1j * (
        kx * projection_k * nonlinear_hat[0] +
        ky * projection_k * nonlinear_hat[1] +
        kz * projection_k * nonlinear_hat[2]
    ) - viscosity * k_squared * velocity_hat

    return rhs

def rk4(f, current_time, y, time_step):
    """
    Fourth-order Runge-Kutta (RK4) method for time integration.

    Parameters:
    f (function): Function representing the time derivative (e.g., compute_rhs).
    current_time (float): Current time in the simulation.
    y (ndarray): Current value of the variable being integrated.
    time_step (float): Time step for the integration.

    Returns:
    ndarray: The integrated value after one RK4 step.
    """
    k1 = f(current_time, y)
    k2 = f(current_time + time_step/2, y + time_step/2 * k1)
    k3 = f(current_time + time_step/2, y + time_step/2 * k2)
    k4 = f(current_time + time_step, y + time_step * k3)
    return time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

# Set up initial conditions for Taylor-Green vortex
x = np.linspace(0, domain_size, grid_points, endpoint=False)
y = np.linspace(0, domain_size, grid_points, endpoint=False)
z = np.linspace(0, domain_size, grid_points, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Define initial velocity field based on Taylor-Green vortex
initial_velocity = np.array([
    np.sin(X) * np.cos(Y) * np.cos(Z),
    -np.cos(X) * np.sin(Y) * np.cos(Z),
    np.zeros_like(X)
])
velocity_hat = fft_transform.fft(initial_velocity)  # Transform the initial velocity field to Fourier space

# Function to plot velocity field in 3D
def plot_velocity_field(velocity, title):
    """
    Plot the 3D velocity field using quiver plots.

    Parameters:
    velocity (ndarray): Velocity field in real space.
    title (str): Title for the plot.
    """
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')
    norm = np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)  # Compute magnitude of velocity
    norm[norm == 0] = 1  # Avoid division by zero for normalization
    
    # Plot velocity vectors
    ax.quiver(X, Y, Z, velocity[0]/norm, velocity[1]/norm, velocity[2]/norm, length=0.1, normalize=True, color='blue', alpha=0.75)
    
    ax.set_xlim([0, domain_size])
    ax.set_ylim([0, domain_size])
    ax.set_zlim([0, domain_size])
    ax.set_title(title, fontsize=14, color='darkred')
    plt.show()

# Plot the initial velocity field
plot_velocity_field(initial_velocity, 'Initial Velocity Field')

# Time-stepping parameters
time_step = 0.01
final_time = 1.0
iterations = int(final_time / time_step)  # Number of iterations for time-stepping

# Main simulation loop
current_time = 0
energy_list = []

for step in range(iterations):
    velocity_hat += rk4(compute_rhs, current_time, velocity_hat, time_step)  # Update velocity using RK4
    current_time += time_step
    
    # Compute energy at each time step
    energy = np.sum(np.abs(velocity_hat)**2) / (2 * grid_points**3)
    energy_list.append(energy)
    
    # Print energy every 10 steps
    if step % 10 == 0:
        print(f"Step {step}, t = {current_time:.3f}, Energy = {energy:.6f}")

# Convert final velocity back to real space
final_velocity = fft_transform.ifft(velocity_hat)

# Compute analytical solution for comparison
analytical_velocity_hat = velocity_hat * np.exp(-2 * viscosity * k_squared * final_time)
analytical_velocity = fft_transform.ifft(analytical_velocity_hat)

# Function to plot initial, final, and analytical velocity fields for comparison
def plot_final_fields(initial_velocity, final_velocity, analytical_velocity):
    """
    Plot the initial, numerical final, and analytical final velocity fields in 3D.

    Parameters:
    initial_velocity (ndarray): Initial velocity field.
    final_velocity (ndarray): Numerical solution for the velocity field at final time.
    analytical_velocity (ndarray): Analytical solution for the velocity field at final time.
    """
    fig = plt.figure(figsize=(18, 6))
    
    norm_init = np.sqrt(initial_velocity[0]**2 + initial_velocity[1]**2 + initial_velocity[2]**2)
    norm_final = np.sqrt(final_velocity[0]**2 + final_velocity[1]**2 + final_velocity[2]**2)
    norm_analytical = np.sqrt(analytical_velocity[0]**2 + analytical_velocity[1]**2 + analytical_velocity[2]**2)
    
    norm_init[norm_init == 0] = 1
    norm_final[norm_final == 0] = 1
    norm_analytical[norm_analytical == 0] = 1

    # Initial velocity field
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.quiver(X, Y, Z, initial_velocity[0]/norm_init, initial_velocity[1]/norm_init, initial_velocity[2]/norm_init, 
               length=0.1, color='green', linewidth=0.7, alpha=0.8)
    ax1.set_xlim([0, domain_size])
    ax1.set_ylim([0, domain_size])
    ax1.set_zlim([0, domain_size])
    ax1.set_title('Initial Velocity Field', fontsize=12, color='darkgreen')

    # Numerical solution at final time
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.quiver(X, Y, Z, final_velocity[0]/norm_final, final_velocity[1]/norm_final, final_velocity[2]/norm_final, 
               length=0.1, color='blue', linewidth=0.7, alpha=0.8)
    ax2.set_xlim([0, domain_size])
    ax2.set_ylim([0, domain_size])
    ax2.set_zlim([0, domain_size])
    ax2.set_title('Numerical Solution at Final Time', fontsize=12, color='darkblue')

    # Analytical solution at final time
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.quiver(X, Y, Z, analytical_velocity[0]/norm_analytical, analytical_velocity[1]/norm_analytical, analytical_velocity[2]/norm_analytical, 
               length=0.1, color='red', linewidth=0.7, alpha=0.8)
    ax3.set_xlim([0, domain_size])
    ax3.set_ylim([0, domain_size])
    ax3.set_zlim([0, domain_size])
    ax3.set_title('Analytical Solution at Final Time', fontsize=12, color='darkred')

    plt.show()

# Plot final velocity fields
plot_final_fields(initial_velocity, final_velocity, analytical_velocity)

# Plot energy over time
plt.figure()
plt.plot(np.linspace(0, final_time, iterations), energy_list, color='purple', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title('Energy of the System Over Time', fontsize=14, color='purple')
plt.grid(True)
plt.show()

print("Simulation complete.")
