import healpy as hp
import numpy as np

def add_noise_to_map(cmb_map, noise_level=0.0005):
    """Add Gaussian noise to a CMB map."""
    noisy_map = cmb_map + np.random.normal(scale=noise_level, size=len(cmb_map))
    return noisy_map

def extract_power_spectrum(cmb_map, lmax=2000):
    """Extract power spectrum from a CMB map."""
    cl = hp.anafast(cmb_map, lmax=lmax)
    return cl

def preprocess_data(cmb_map, noise_level=0.0005):
    """Preprocess CMB data by adding noise and extracting spectrum."""
    noisy_map = add_noise_to_map(cmb_map, noise_level)
    cl = extract_power_spectrum(noisy_map)
    return noisy_map, cl
