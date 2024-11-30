import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def map_to_frequency(value, min_val, max_val, min_freq, max_freq):
    """Map a value to a frequency range."""
    return min_freq + (max_freq - min_freq) * (value - min_val) / (max_val - min_val)

def generate_sound_wave(frequency, amplitude, duration, sample_rate=44100):
    """Generate a sound wave for a given frequency and amplitude."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave

def luminosity_to_sound(luminosity, redshift, faint_end_slope, duration=5, sample_rate=44100):
    """Convert luminosity function into a harmonic soundscape."""
    num_points = len(luminosity)
    sound_wave = np.zeros(int(sample_rate * duration))
    time_step = duration / num_points

    for i in range(num_points):
        freq = map_to_frequency(redshift[i], min(redshift), max(redshift), 200, 2000)
        amp = luminosity[i] / max(luminosity)
        wave = generate_sound_wave(freq, amp, time_step, sample_rate)
        sound_wave[i * len(wave):(i + 1) * len(wave)] += wave

    # Normalize sound wave
    sound_wave = np.int16(sound_wave / np.max(np.abs(sound_wave)) * 32767)
    return sound_wave

def visualize_harmonics(luminosity, redshift):
    """Perform Fourier Transform and visualize harmonic components."""
    ft = fft(luminosity)
    freqs = np.fft.fftfreq(len(luminosity), d=(redshift[1] - redshift[0]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, np.abs(ft))
    plt.title("Fourier Transform of Luminosity Function")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# Example data (replace with real luminosity and redshift data)
luminosity = np.array([10, 20, 15, 30, 25, 10])
redshift = np.linspace(0.02, 0.5, len(luminosity))
faint_end_slope = -1.5  # Example faint-end slope

# Generate and save the soundscape
sound_wave = luminosity_to_sound(luminosity, redshift, faint_end_slope)
write("luminosity_soundscape.wav", 44100, sound_wave)

# Visualize harmonic components
visualize_harmonics(luminosity, redshift)
