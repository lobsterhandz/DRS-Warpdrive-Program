import healpy as hp
import numpy as np
import camb
import matplotlib.pyplot as plt

def generate_theoretical_spectra(lmax=2000):
    """Generate theoretical CMB power spectra using CAMB."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    
    results = camb.get_results(pars)
    cl_theory = results.get_cmb_power_spectra()['total'][:, 0]  # TT spectrum
    return cl_theory

def simulate_cmb_map(cl, nside=512):
    """Simulate a CMB map given a power spectrum."""
    cmb_map = hp.synfast(cl, nside=nside, verbose=False)
    return cmb_map

def plot_cmb_map(cmb_map, title="CMB Map"):
    """Plot a CMB map."""
    hp.mollview(cmb_map, title=title, unit="K", cmap="coolwarm")
    plt.show()
