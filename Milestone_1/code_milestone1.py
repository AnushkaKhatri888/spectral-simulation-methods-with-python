import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import muFFT
import muGrid

class FFTTransformer:
    def __init__(self, nb_grid_pts, engine='pocketfft'):
        """
        Initialize the FFTTransformer class.

        Parameters:
        nb_grid_pts (tuple): Number of grid points in each direction (x, y, z).
        engine (str): FFT engine to be used (default is 'pocketfft').
        """
        self.nb_grid_pts = nb_grid_pts
        self.engine = engine
        self.comm = muGrid.Communicator()  # Initialize the communicator
        self.fft = muFFT.FFT(nb_grid_pts, engine=self.engine, communicator=self.comm)  # Initialize the FFT processor

    def fft_forward(self, real_field_data):
        """
        Perform forward FFT to convert a real-space field to Fourier space.

        Parameters:
        real_field_data (ndarray): Real-space field data.

        Returns:
        ndarray: Fourier-space representation of the field.
        """
        real_field = self.fft.real_space_field('real_field')  # Define real-space field
        real_field.p = real_field_data  # Assign input data to real-space field
        fourier_field = self.fft.fourier_space_field('fourier_field')  # Define Fourier-space field
        self.fft.fft(real_field, fourier_field)  # Perform FFT
        return fourier_field.p  # Return Fourier-space data

    def fft_backward(self, fourier_field_data):
        """
        Perform inverse FFT to convert Fourier-space data back to real space.

        Parameters:
        fourier_field_data (ndarray): Fourier-space field data.

        Returns:
        ndarray: Real-space representation of the field.
        """
        fourier_field = self.fft.fourier_space_field('fourier_field')  # Define Fourier-space field
        fourier_field.p = fourier_field_data  # Assign input data to Fourier-space field
        real_field = self.fft.real_space_field('real_field')  # Define real-space field
        self.fft.ifft(fourier_field, real_field)  # Perform inverse FFT
        return real_field.p  # Return real-space data


def curl(u_cxyz):
    """
    Compute the curl of a 3D vector field using FFT.

    Parameters:
    u_cxyz (ndarray): 3D vector field in real space.

    Returns:
    ndarray: Curl of the input vector field.
    """
    grid_shape = u_cxyz.shape[:-1]  # Extract the shape of the grid (excluding the vector components)
    fft_transformer = FFTTransformer(grid_shape)  # Initialize FFT transformer

    # Forward FFT for each component of the vector field
    field_shape = fft_transformer.fft_forward(u_cxyz[..., 0]).shape  # Get the shape of the Fourier-transformed field
    u_hat = np.zeros(field_shape + (3,), dtype=np.complex128)  # Initialize Fourier-space vector field

    # Perform FFT on each component of the vector field
    for i in range(3):
        u_hat[..., i] = fft_transformer.fft_forward(u_cxyz[..., i])

    # Create wave number vectors for each direction
    kx = np.fft.fftfreq(field_shape[0]) * 2j * np.pi
    ky = np.fft.fftfreq(field_shape[1]) * 2j * np.pi
    kz = np.fft.fftfreq(field_shape[2]) * 2j * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

    # Calculate curl in Fourier space
    curl_hat = np.zeros_like(u_hat, dtype=np.complex128)  # Initialize Fourier-space curl
    curl_hat[..., 0] = ky * u_hat[..., 2] - kz * u_hat[..., 1]  # Curl x-component
    curl_hat[..., 1] = kz * u_hat[..., 0] - kx * u_hat[..., 2]  # Curl y-component
    curl_hat[..., 2] = kx * u_hat[..., 1] - ky * u_hat[..., 0]  # Curl z-component

    # Perform inverse FFT to convert curl back to real space
    curl_real = np.zeros_like(u_cxyz, dtype=np.float64)  # Initialize real-space curl

    # Inverse FFT on each component of the curl
    for i in range(3):
        curl_real[..., i] = fft_transformer.fft_backward(curl_hat[..., i]).real

    return curl_real  # Return the real-space curl


def test_constant_field():
    """
    Test the curl function by using a constant field. The curl of a constant field should be zero.
    """
    nb_grid_pts = (32, 32, 2)  # Define grid size
    constant_field = np.ones([*nb_grid_pts, 3])  # Create a constant vector field
    curl_result = curl(constant_field)  # Compute the curl
    np.testing.assert_allclose(curl_result, 0, atol=1e-10)  # Verify that the curl is zero
    print("Test passed: Constant field results in a zero curl.")


def generate_vector_field_and_compute_curl():
    """
    Generate a sample vector field and compute its curl.

    Returns:
    tuple: Grid coordinates, generated vector field, and computed curl.
    """
    nb_grid_pts = (32, 32, 2)  # Define grid size

    # Generate grid coordinates
    coords = np.array(np.meshgrid(
        np.linspace(0, 1, nb_grid_pts[0]),
        np.linspace(0, 1, nb_grid_pts[1]),
        np.linspace(0, 1, nb_grid_pts[2]),
        indexing='ij'
    ))

    # Define a vector field using a cross-product formula
    norm = np.array([0, 0, 1])
    vector_field = np.zeros(coords.shape[1:] + (3,))
    vector_field[..., 0] = norm[1] * (coords[2] - 0.5) - norm[2] * (coords[1] - 0.5)
    vector_field[..., 1] = norm[2] * (coords[0] - 0.5) - norm[0] * (coords[2] - 0.5)
    vector_field[..., 2] = norm[0] * (coords[1] - 0.5) - norm[1] * (coords[0] - 0.5)

    curl_result = curl(vector_field)  # Compute the curl
    return coords, vector_field, curl_result  # Return the grid, field, and curl


def plot_vector_field_and_curl(coords, vector_field, curl_result):
    """
    Plot and visualize the generated vector field and its curl.

    Parameters:
    coords (ndarray): Grid coordinates.
    vector_field (ndarray): Generated vector field.
    curl_result (ndarray): Computed curl of the vector field.
    """
    fig = plt.figure(figsize=(14, 6))

    # Plot the vector field
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.quiver(coords[0], coords[1], coords[2], vector_field[..., 0], vector_field[..., 1], vector_field[..., 2],
               length=0.1, normalize=True, color='orange', alpha=0.6, linewidth=0.8)
    ax1.set_title('Vector Field')

    # Plot the curl of the vector field
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(coords[0], coords[1], coords[2], curl_result[..., 0], curl_result[..., 1], curl_result[..., 2],
               length=0.1, normalize=True, color='grey', alpha=0.8, linewidth=0.8)
    ax2.set_title('Curl of the Vector Field')

    plt.show()


def display_sample_curl_values(curl_result):
    """
    Display sample values of the computed curl at various grid points.

    Parameters:
    curl_result (ndarray): Computed curl of the vector field.
    """
    print("Sample curl values at various grid points:")
    for i in range(0, curl_result.shape[0], max(1, curl_result.shape[0] // 5)):
        for j in range(0, curl_result.shape[1], max(1, curl_result.shape[1] // 5)):
            for k in range(0, curl_result.shape[2], max(1, curl_result.shape[2] // 2)):
                print(f"Curl at ({i}, {j}, {k}): {curl_result[i, j, k]}")


def test_nonzero_curl():
    """
    Test and visualize the curl of a non-zero vector field.
    """
    print("Testing non-zero curl field")
    nb_grid_pts = (32, 32, 2)  # Define grid size

    # Generate grid coordinates
    norm = np.array([0, 0, 1])
    coords = np.array(np.meshgrid(
        np.linspace(0, 1, nb_grid_pts[0]),
        np.linspace(0, 1, nb_grid_pts[1]),
        np.linspace(0, 1, nb_grid_pts[2]),
        indexing='ij'
    ))

    # Generate a vector field using the cross-product formula
    vector_field = np.cross(norm, coords - 0.5, axis=0)
    vector_field = np.moveaxis(vector_field, 0, -1)

    # Compute the curl of the vector field
    curl_result = curl(vector_field)

    # Visualize the vector field and its curl
    plot_vector_field_and_curl(coords, vector_field, curl_result)

    # Display sample curl values
    display_sample_curl_values(curl_result)

    # Calculate and print the mean of the curl
    mean_curl = np.mean(curl_result, axis=(0, 1, 2))
    print(f"Mean curl: {mean_curl}")


# Run tests
test_constant_field()
test_nonzero_curl()
