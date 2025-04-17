import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from DataLoader import DataLoader as DL
import os, re, glob
from DataLoader import BeamAnalysis as BA
from DataLoader import DataLoader as DL


exp_directory = 'C:/Users/felnor/OneDrive - Excillum AB/Desktop/new_analysis/angular_analysis/29-03-25/experimental_data'

filename = os.path.join(exp_directory, 'ESF_HR_nofilter.npy')
# load the data array
data = np.load(filename)
print(f"loaded data shape = {data.shape}")

# styling
sns.set_context('talk')
sns.set_style('whitegrid')

plt.figure(figsize=(14, 8))
lim = 10e-6

# For FWHM calculation and collecting results
fwhm_data = []

# Option to select specific voltages or just one of each
voltage_selection_mode = 'all'  # Options: 'one_of_each', 'specific'
specific_voltages = [80, 90]  # Only used if mode is 'specific' - specify kV values

# Initialize set to track processed voltages
processed_voltages = set()

# Function to determine if we should process this voltage
def should_process_voltage(voltage):
    if voltage_selection_mode == 'all':
        if voltage in processed_voltages:
            return False
        processed_voltages.add(voltage)
        return True
    elif voltage_selection_mode == 'specific':
        if voltage in specific_voltages and voltage not in processed_voltages:
            processed_voltages.add(voltage)
            return True
        return False
    return True  # Default case

for row_idx in range(data.shape[0]):
    row = data[row_idx]
    
    # unpack the metadata
    acc_voltage = row[0]/1e3          # [kV]
    
    # Skip if we shouldn't process this voltage
    if not should_process_voltage(acc_voltage):
        continue
    
    spot_setting = row[1]         # (µm)
    x_start = row[2]              # mm
    x_stop = row[3]               # mm
    N_points = int(row[4])        # length of ESF
    
    # extract the ESF values (they live in the last N_points entries)
    esf = row[-N_points:]
    
    # build the x-axis (evenly spaced from x_start → x_stop)
    x = np.linspace(x_start, x_stop, N_points)
    
    # compute the PSF = d(ESF)/dx
    psf = np.gradient(esf, x)
    psf /= np.max(psf)  # normalize to max value
    
    # calculate FWHM
    fwhm = BA.calculate_fwhm(x, psf) * 1e6
    fwhm_data.append({
        'Acceleration Voltage (kV)': acc_voltage,
        'Spot Setting (µm)': spot_setting * 1e6,
        'FWHM (µm)': fwhm
    })
    
    # plot with label showing voltage, spot setting and FWHM
    plt.plot(x, psf, lw=1.5, label=f"{acc_voltage} kV, Spot {spot_setting} µm, FWHM = {fwhm:.2f} µm")

# Create DataFrame with FWHM results
fwhm_df = pd.DataFrame(fwhm_data)
print(fwhm_df)

plt.xlabel("Position (µm)")
plt.ylabel("Point‐Spread (PSF)")
plt.ylim(-0.1,1.1)
plt.xlim(-lim, lim)
plt.title(f"PSF Comparison with FWHM Measurements")
plt.legend()
plt.tight_layout()
plt.show()
