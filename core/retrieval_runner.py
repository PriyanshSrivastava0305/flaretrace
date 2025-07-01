import numpy as np
import matplotlib.pyplot as plt
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import M_jup, R_jup, R_sun

def run_platon_retrieval(
    wavelength_um,
    flux=None,
    star_radius=1.0,
    planet_radius=1.0,
    planet_mass=1.0,
    temp=1000,
    logZ=0,
    CO_ratio=0.53
):
    """
    Compute theoretical transit depths using PLATON's TransitDepthCalculator.

    Parameters:
    - wavelength_um: 1D array of wavelengths in microns
    - flux: Optional flux (to be plotted later for comparison)
    - star_radius, planet_radius in units of solar/jupiter radius
    - temp in Kelvin

    Returns:
    - wavelengths (μm), model_depths, info_dict
    """
    calculator = TransitDepthCalculator()

    Rs = star_radius * R_sun
    Rp = planet_radius * R_jup
    Mp = planet_mass * M_jup
    T = temp

    wavelengths_m = np.array(wavelength_um) * 1e-6  # Convert μm to meters

    wavelengths, model_depths, info_dict = calculator.compute_depths(
        Rs, Mp, Rp, T,
        logZ=logZ,
        CO_ratio=CO_ratio,
        full_output=True,
        wavelengths=wavelengths_m  # use observed bins
    )

    return wavelengths * 1e6, model_depths, info_dict  # Convert back to μm for plotting


def plot_fit(wavelength_um, original_flux, corrected_flux=None, model_wavelengths=None, model_depths=None):
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength_um, original_flux, label="Observed", alpha=0.7, marker='o')

    if corrected_flux is not None:
        plt.plot(wavelength_um, corrected_flux, label="Corrected", alpha=0.7, marker='o')

    if model_wavelengths is not None and model_depths is not None:
        plt.plot(model_wavelengths, model_depths, label="PLATON Model", linestyle="--")

    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Transit Depth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
