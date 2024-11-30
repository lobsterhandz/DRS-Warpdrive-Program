import numpy as np
from scipy.io.wavfile import write


# Refined Oscillatory Tensor (T_osc)
def T_osc(t, x, frequencies, amplitudes, stochastic_noise=0.01):
    """
    Compute the refined oscillatory tensor based on harmonic frequencies and amplitudes.

    Args:
        t (numpy.ndarray): Time array.
        x (numpy.ndarray): Spatial array.
        frequencies (list): List of dominant frequencies.
        amplitudes (list): Corresponding amplitudes for frequencies.
        stochastic_noise (float): Amplitude of stochastic noise.

    Returns:
        numpy.ndarray: Oscillatory tensor (time, space).
    """
    time, space = np.meshgrid(t, x, indexing="ij")
    harmonic_sum = np.sum(
        [amp * np.sin(2 * np.pi * freq * time + space) for freq, amp in zip(frequencies, amplitudes)],
        axis=0
    )
    noise = stochastic_noise * np.random.normal(0, 1, harmonic_sum.shape)
    return harmonic_sum + noise


# Refined Resonance Tensor (T_res)
def T_res(t, x, base_frequency, amplitude, damping=0.1, frequencies=[]):
    """
    Compute the refined resonance tensor based on harmonic interference and damping.

    Args:
        t (numpy.ndarray): Time array.
        x (numpy.ndarray): Spatial array.
        base_frequency (float): Base resonance frequency.
        amplitude (float): Amplitude of resonance.
        damping (float): Damping factor for resonance.
        frequencies (list): List of frequencies for interference patterns.

    Returns:
        numpy.ndarray: Resonance tensor (time, space).
    """
    time, space = np.meshgrid(t, x, indexing="ij")
    interference = np.sum(
        [
            np.sin(2 * np.pi * f1 * time) * np.sin(2 * np.pi * f2 * time)
            for i, f1 in enumerate(frequencies)
            for j, f2 in enumerate(frequencies)
            if i != j
        ],
        axis=0,
    )
    resonance = amplitude * np.cos(2 * np.pi * base_frequency * time) * np.exp(-damping * time)
    return resonance + interference


# Generate Sonified Waveform
def generate_sonified_wave(frequencies, amplitudes, duration=5, sample_rate=44100):
    """
    Generate and save a sonified waveform based on harmonic frequencies.

    Args:
        frequencies (list): List of dominant frequencies.
        amplitudes (list): Corresponding amplitudes for frequencies.
        duration (float): Duration of the audio in seconds.
        sample_rate (int): Audio sample rate.

    Returns:
        None
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.sum([amp * np.sin(2 * np.pi * freq * t) for freq, amp in zip(frequencies, amplitudes)], axis=0)
    waveform = (waveform / np.max(np.abs(waveform)) * 32767).astype(np.int16)  # Normalize to 16-bit PCM
    write("cosmic_harmonics.wav", sample_rate, waveform)
    print("Sonification complete. File saved as 'cosmic_harmonics.wav'.")


# Example Frequencies and Amplitudes from Fourier Transform
frequencies = [0.5, 1.0, 1.5, 2.0]  # Dominant frequencies
amplitudes = [1.0, 0.8, 0.6, 0.4]   # Corresponding amplitudes

# Generate and Save Sonified Audio
generate_sonified_wave(frequencies, amplitudes)

# Time and Space Arrays
t = np.linspace(0, 10, 1000)  # Time array
x = np.linspace(0, 2 * np.pi, 100)  # Spatial array

# Generate Oscillatory and Resonance Tensors
osc_tensor = T_osc(t, x, frequencies, amplitudes)
res_tensor = T_res(t, x, base_frequency=0.5, amplitude=1.0, damping=0.1, frequencies=frequencies)

# Display Results
print("Refined oscillatory tensor (T_osc) generated with shape:", osc_tensor.shape)
print("Refined resonance tensor (T_res) generated with shape:", res_tensor.shape)
print("Ready for further validation and visualization.")
