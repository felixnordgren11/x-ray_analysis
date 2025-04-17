import os
import uproot as ur
import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib.pyplot as plt
import awkward as ak
import sklearn.preprocessing as skp
import seaborn as sns
import time
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import linregress
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import glob
import re

sns.set_theme()

# ssh felix@wexc006 
# Define the Gaussian function
class BeamAnalysis():

    def __init__(self):
        pass

    def gaussian(x, mu, std, amplitude):
        return amplitude * np.exp(-(x - mu)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
    
    # Define Voigt function (Gaussian + Lorentzian convolution)
    def voigt(x, mu, sigma, gamma, amplitude):
        """Voigt function using scipy's wofz (Faddeeva function)."""
        z = (x - mu + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(sp.wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    def lorentz(x, mu, gamma, amplitude):
        ### Lorentz curve function
        return amplitude * gamma / (np.pi * ((x - mu)**2 + gamma**2))
    

    def make_gaussian_kernel(size, mean, std): # for convolution
        # Generate an integer array from -size//2 ... +size//2
        x = np.arange(size) - (size // 2)
        # Evaluate Gaussian at each point
        kernel = np.exp(-((x - mean)**2) / (2 * std**2))
        # Normalize so max is 1
        kernel /= np.max(kernel)
        return kernel
    
    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y):
        x, y = xy
        return amplitude * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))


    def relative_power(data, left, right):
        return integrate.simpson(data[left:right], dx=1)/integrate.simpson(data, dx=1)

    

    # Define the Freedman-Diaconis binning rule
    def optimal_bins(data):
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        bin_width = 2 * (iqr) * len(data) ** (-1/3)
        bins = int((data.max() - data.min()) / bin_width)
        return max(bins, 10)  # Ensure at least 10 bins
    
    def twoD_Gaussian(amp0, x0, y0, amp1=13721, x1=356, y1=247, amp2=14753, x2=291,  y2=339, sigma=40):
        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        return lambda x, y:  (amp0*np.exp(-(((x0-x)/sigma)**2+((y0-y)/sigma)**2)/2))+(
                             amp1*np.exp(-(((x1-x)/sigma)**2+((y1-y)/sigma)**2)/2))+(
                             amp2*np.exp(-(((x2-x)/sigma)**2+((y2-y)/sigma)**2)/2))

    def twoD_GaussianCF(xy, amp0, x0, y0, amp1=13721, amp2=14753, x1=356, y1=247, x2=291,  y2=339, sigma_x=12, sigma_y=12):

        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        
        x, y = xy
        
        g = (amp0*np.exp(-(((x0-x)/sigma_x)**2+((y0-y)/sigma_y)**2)/2))+(
            amp1*np.exp(-(((x1-x)/sigma_x)**2+((y1-y)/sigma_y)**2)/2))+(
            amp2*np.exp(-(((x2-x)/sigma_x)**2+((y2-y)/sigma_y)**2)/2))

        return g.ravel()
    


class DataLoader():
    def __init__(self):
        pass

    
    def load_data(filename, axis: str = 'x', material=None, interaction=None, splitting_factor_brem=200, splitting_factor_char=1, ):

            '''

            Generates positional data in mm - x, y, z, xt (projected x) - from a root file.

            If you choose axis = 'ene', the function will return the energy of the (weighted) particles instead.

            The output data is a vector, where the ith element of the vector is the position (in 'axis' direction) in the target material that the ith photon was generated.
            Creating a histogram from the data will give the distribution of where photons were generated in the target material, and hence a good estimate of the spot size.
            To get the correct spot size in x direction at a detector angle =/= 0, transform the x data as xt = x * cos(angle) - z * sin(angle).


            The data can be filtered by interaction type (brem or char) and material (W or Di).
            To convert to µm just multiply by 1e3.

            '''
            file = ur.open(filename)
            tree = file["sim_res_out;1"]
            types = ["Particle_gen_proj_x", "Particle_gen_x", "Particle_gen_z", "Particle_gen_y", "Volume_gen", "Event_weight", "ene", "Particle_energy_out"]
            data = tree.arrays(types, library="ak")
            w, di = data["Volume_gen"] == "Tungsten", data["Volume_gen"] == "Diamond"
            selection_brem, selection_char = data["Event_weight"] == 1/splitting_factor_brem, data["Event_weight"] == 1/splitting_factor_char # Change if using other splitting factors. Weight = 1/(splitting factor)
            # in keV/kV whatever you want to call it
            tree2 = file["info;1"]
            types2 = ["Diamond_thick_um", "Film_thick_um", "Det_Ang_deg", "Nsim", "Time_elapsed"]
            data2 = tree2.arrays(types2, library="ak")

            time = np.sum(ak.to_numpy(data2["Time_elapsed"])) # time elapsed in s ( summed over all threads )
            acc_voltage = data["ene"][0]
            angle = float(ak.to_numpy(data2["Det_Ang_deg"][0])) # detection angle
            diamond_thickness = ak.to_numpy(data2["Diamond_thick_um"][0])
            film_thickness = ak.to_numpy(data2["Film_thick_um"][0])
            n_primaries = ak.to_numpy(data2["Nsim"][0]) # number of primary evts

            x_tot_brem = ak.to_numpy(data["Particle_gen_x"][selection_brem])
            x_tot_char = ak.to_numpy(data["Particle_gen_x"][selection_char])
            x_W_brem   = ak.to_numpy(data["Particle_gen_x"][w & selection_brem])
            x_W_char   = ak.to_numpy(data["Particle_gen_x"][w & selection_char])
            x_Di_brem  = ak.to_numpy(data["Particle_gen_x"][di & selection_brem])
            x_Di_char  = ak.to_numpy(data["Particle_gen_x"][di & selection_char])

            xt_tot_brem = ak.to_numpy(data["Particle_gen_proj_x"][selection_brem])
            xt_tot_char = ak.to_numpy(data["Particle_gen_proj_x"][selection_char])
            xt_W_brem   = ak.to_numpy(data["Particle_gen_proj_x"][w & selection_brem])
            xt_W_char   = ak.to_numpy(data["Particle_gen_proj_x"][w & selection_char])
            xt_Di_brem  = ak.to_numpy(data["Particle_gen_proj_x"][di & selection_brem])
            xt_Di_char  = ak.to_numpy(data["Particle_gen_proj_x"][di & selection_char])

            y_tot_brem = ak.to_numpy(data["Particle_gen_y"][selection_brem])
            y_tot_char = ak.to_numpy(data["Particle_gen_y"][selection_char])
            y_W_brem   = ak.to_numpy(data["Particle_gen_y"][w & selection_brem])
            y_W_char   = ak.to_numpy(data["Particle_gen_y"][w & selection_char])
            y_Di_brem  = ak.to_numpy(data["Particle_gen_y"][di & selection_brem])
            y_Di_char  = ak.to_numpy(data["Particle_gen_y"][di & selection_char])

            z_tot_brem = ak.to_numpy(data["Particle_gen_z"][selection_brem])
            z_tot_char = ak.to_numpy(data["Particle_gen_z"][selection_char])
            z_W_brem   = ak.to_numpy(data["Particle_gen_z"][w & selection_brem])
            z_W_char   = ak.to_numpy(data["Particle_gen_z"][w & selection_char])
            z_Di_brem  = ak.to_numpy(data["Particle_gen_z"][di & selection_brem])
            z_Di_char  = ak.to_numpy(data["Particle_gen_z"][di & selection_char])

            weights_tot_brem = ak.to_numpy(data["Event_weight"][selection_brem])
            weights_tot_char = ak.to_numpy(data["Event_weight"][selection_char])
            weights_W_brem   = ak.to_numpy(data["Event_weight"][w & selection_brem])
            weights_W_char   = ak.to_numpy(data["Event_weight"][w & selection_char])
            weights_Di_brem  = ak.to_numpy(data["Event_weight"][di & selection_brem])
            weights_Di_char  = ak.to_numpy(data["Event_weight"][di & selection_char])

            # Extract energy information
            energies_tot_brem = ak.to_numpy(data["Particle_energy_out"][selection_brem])
            energies_tot_char = ak.to_numpy(data["Particle_energy_out"][selection_char])
            energies_W_brem = ak.to_numpy(data["Particle_energy_out"][w & selection_brem])
            energies_W_char = ak.to_numpy(data["Particle_energy_out"][w & selection_char])
            energies_Di_brem = ak.to_numpy(data["Particle_energy_out"][di & selection_brem])
            energies_Di_char = ak.to_numpy(data["Particle_energy_out"][di & selection_char])

            '''
            Weights can be used with the histogram function to get the weighted histogram if particle splitting has been used
            '''

            weights_tot = np.concatenate((weights_tot_brem, weights_tot_char)) 
            weights_W = np.concatenate((weights_W_brem, weights_W_char))
            weights_Di = np.concatenate((weights_Di_brem, weights_Di_char))

            sim_parameters = pd.DataFrame({
                'acc_voltage': acc_voltage,
                'angle': angle,
                'diamond_thickness': diamond_thickness,
                'tungsten_thickness': film_thickness,
                'n_primaries': n_primaries,
                'simulation_time': time,
                'splitting_factor_brem': splitting_factor_brem,
                })


            if axis == 'x':
                if material is None:
                    if interaction is None:
                        x_tot = np.concatenate((x_tot_brem, x_tot_char))
                        weights = weights_tot
                    elif interaction == 'brem':
                        x_tot = x_tot_brem
                        weights = weights_tot_brem
                    elif interaction == 'char':
                        x_tot = x_tot_char
                        weights = weights_tot_char
                elif material == 'W':
                    if interaction is None:
                        x_tot = np.concatenate((x_W_brem, x_W_char))
                        weights = weights_W
                    elif interaction == 'brem':
                        x_tot = x_W_brem
                        weights = weights_W_brem
                    elif interaction == 'char':
                        x_tot = x_W_char
                        weights = weights_W_char
                elif material == 'Di':
                    if interaction is None:
                        x_tot = np.concatenate((x_Di_brem, x_Di_char))
                        weights = weights_Di
                    elif interaction == 'brem':
                        x_tot = x_Di_brem
                        weights = weights_Di_brem
                    elif interaction == 'char':
                        x_tot = x_Di_char
                        weights = weights_Di_char

                x_tot = x_tot - np.sin(np.radians(angle))*diamond_thickness/1e3 - np.sin(np.radians(angle))*film_thickness/1e3
                return x_tot, weights, sim_parameters


            if axis == 'y':
                if material is None:
                    if interaction is None:
                        y_tot = np.concatenate((y_tot_brem, y_tot_char))
                        weights = weights_tot
                    elif interaction == 'brem':
                        y_tot = y_tot_brem
                        weights = weights_tot_brem
                    elif interaction == 'char':
                        y_tot = y_tot_char
                        weights = weights_tot_char
                elif material == 'W':
                    if interaction is None:
                        y_tot = np.concatenate((y_W_brem, y_W_char))
                        weights = weights_W
                    elif interaction == 'brem':
                        y_tot = y_W_brem
                        weights = weights_W_brem
                    elif interaction == 'char':
                        y_tot = y_W_char
                        weights = weights_W_char
                elif material == 'Di':
                    if interaction is None:
                        y_tot = np.concatenate((y_Di_brem, y_Di_char))
                        weights = weights_Di
                    elif interaction == 'brem':
                        y_tot = y_Di_brem
                        weights = weights_Di_brem
                    elif interaction == 'char':
                        y_tot = y_Di_char
                        weights = weights_Di_char
                return y_tot, weights, sim_parameters


            if axis == 'z':
                if material is None:
                    if interaction is None:
                        z_tot = np.concatenate((z_tot_brem, z_tot_char))
                        weights = weights_tot
                    elif interaction == 'brem':
                        z_tot = z_tot_brem
                        weights = weights_tot_brem
                    elif interaction == 'char':
                        z_tot = z_tot_char
                        weights = weights_tot_char
                elif material == 'W':
                    if interaction is None:
                        z_tot = np.concatenate((z_W_brem, z_W_char))
                        weights = weights_W
                    elif interaction == 'brem':
                        z_tot = z_W_brem
                        weights = weights_W_brem
                    elif interaction == 'char':
                        z_tot = z_W_char
                        weights = weights_W_char
                elif material == 'Di':
                    if interaction is None:
                        z_tot = np.concatenate((z_Di_brem, z_Di_char))
                        weights = weights_Di
                    elif interaction == 'brem':
                        z_tot = z_Di_brem
                        weights = weights_Di_brem
                    elif interaction == 'char':
                        z_tot = z_Di_char
                        weights = weights_Di_char
                return z_tot, weights, sim_parameters

            if axis == 'xt':
                if material is None:
                    if interaction is None:
                        xt_tot = np.concatenate((xt_tot_brem, xt_tot_char))
                        weights = weights_tot
                    elif interaction == 'brem':
                        xt_tot = xt_tot_brem
                        weights = weights_tot_brem
                    elif interaction == 'char':
                        xt_tot = xt_tot_char
                        weights = weights_tot_char
                elif material == 'W':
                    if interaction is None:
                        xt_tot = np.concatenate((xt_W_brem, xt_W_char))
                        weights = weights_W
                    elif interaction == 'brem':
                        xt_tot = xt_W_brem
                        weights = weights_W_brem
                    elif interaction == 'char':
                        xt_tot = xt_W_char
                        weights = weights_W_char
                elif material == 'Di':
                    if interaction is None:
                        xt_tot = np.concatenate((xt_Di_brem, xt_Di_char))
                        weights = weights_Di
                    elif interaction == 'brem':
                        xt_tot = xt_Di_brem
                        weights = weights_Di_brem
                    elif interaction == 'char':
                        xt_tot = xt_Di_char
                        weights = weights_Di_char
                xt_tot = xt_tot - np.sin(np.radians(angle))*diamond_thickness/1e3 - np.sin(np.radians(angle))*film_thickness/1e3
                return xt_tot, weights, sim_parameters

            if axis == 'ene':
                if material is None:
                    if interaction is None:
                        energies_tot = np.concatenate((energies_tot_brem, energies_tot_char))
                        weights = weights_tot
                    elif interaction == 'brem':
                        energies_tot = energies_tot_brem
                        weights = weights_tot_brem
                    elif interaction == 'char':
                        energies_tot = energies_tot_char
                        weights = weights_tot_char
                elif material == 'W':
                    if interaction is None:
                        energies_tot = np.concatenate((energies_W_brem, energies_W_char))
                        weights = weights_W
                    elif interaction == 'brem':
                        energies_tot = energies_W_brem
                        weights = weights_W_brem
                    elif interaction == 'char':
                        energies_tot = energies_W_char
                        weights = weights_W_char
                elif material == 'Di':
                    if interaction is None:
                        energies_tot = np.concatenate((energies_Di_brem, energies_Di_char))
                        weights = weights_Di
                    elif interaction == 'brem':
                        energies_tot = energies_Di_brem
                        weights = weights_Di_brem
                    elif interaction == 'char':
                        energies_tot = energies_Di_char
                        weights = weights_Di_char
                return energies_tot, weights, sim_parameters
            return None
    







'''
# To get a 2d spot, you can call the function two times, once for x and once for y, and then use the 2d histogram function to get the 2d histogram.
# For example:
# Path to data files directory
data_dir = "angular_analysis/29-03-25/data_files"
# Get one file from the directory
files = glob.glob(os.path.join(data_dir, "*.root"))
if not files:
    print(f"No .root files found in {data_dir}")
    exit()
# Use the first file
sample_file = files[0]
print(f"Using file: {sample_file}")
# Get x and y data from the file
x_data, x_weights, sim_params = load_data(sample_file, axis='x', interaction='brem')
y_data, y_weights, _ = DataLoader.load_data(sample_file, axis='y', interaction='brem')
# Create 2D histogram
hist_range = [[-0.02, 0.02], [-0.02, 0.02]]  # Range in mm
nbins = 1000
bins = [nbins, nbins]  # Number of bins in x and y
hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins, range=hist_range, weights=x_weights)
hist = hist.T  # Transpose to match correct orientation
# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# Plot 2D histogram
im = ax.imshow(hist, extent=extent, origin='lower', aspect='equal', cmap='viridis')
cbar = plt.colorbar(im, ax=ax, label='Counts')
# Add title and labels
title = f"2D Spot - {os.path.basename(sample_file)}"
subtitle = f"{sim_params['acc_voltage']:.0f}kV, {sim_params['angle']}°, Diamond: {sim_params['diamond_thickness']}μm"
ax.set_title(f"{title}\n{subtitle}")
ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
lim = 0.002 #mm
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
plt.grid(False)
# Save and show
plt.tight_layout()
#plt.savefig(f"2d_spot_{os.path.basename(sample_file).replace('.root', '')}.png", dpi=300)
plt.show()
'''


