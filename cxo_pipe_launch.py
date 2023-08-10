import os
import time
from termcolor import colored
import subprocess as sp


import python.cxo_pipe_preproc as prepro
import python.cxo_pipe_spec as spec
import python.cxo_pipe_icm as icm
import python.cxo_pipe_plot as plt


from param import *

ti = time.time()


print("------------------------------------------------------------")
print(colored("Pre-processing", "cyan", None, ["bold"]))
print("------------------------------------------------------------")
res_dir = os.environ["CXO_RES_DIR"] + source_name + "/"
# Import data
prepro.import_data(obsids, res_dir)
# Reprocess data
prepro.reprocess_data(obsids, res_dir)
# Apply energy cuts
prepro.energy_cut(obsids, res_dir)
# Remove flares
prepro.remove_flares(obsids, res_dir)
# Compute maps of the observations
if multiobs:
    prepro.reproj_obsids(obsids, res_dir)
else:
    prepro.make_images(obsids, res_dir)
    prepro.make_psf_map(res_dir)
# Find point sources
is_sources = prepro.find_sources(res_dir, multiobs)
# Check point source regions
prepro.check_sources(res_dir, is_sources)
# Compute a background region for each point source
prepro.bkg_point_sources(res_dir, is_sources)
# Subtract point sources
prepro.subtract_point_sources(res_dir, is_sources, multiobs, obsids)
# Find X-ray peak and centroid locations
Xdepro, Ydepro = prepro.find_peak_cent(res_dir, z, R500, use_peak, fixed_coord)
# Define background region for X-ray surface brightness and spectra
bkg_area = prepro.bkg_region(res_dir, z, R500, Xdepro, Ydepro, multiobs, obsids)
# Define annuli for the surface brightness profile
prepro.find_SB_annuli(res_dir, Xdepro, Ydepro, bkg_area, z, R500, fast_annuli, obsids)
# Compute weights to take vignetting into account
prepro.vignetting_prof(res_dir, obsids)
# Compute vignetting-corrected surface brightness profile
prepro.X_ray_SB_profile(res_dir, obsids, z)

print("------------------------------------------------------------")
print(colored("Spectral analysis", "cyan", None, ["bold"]))
print("------------------------------------------------------------")
# Find the stowed background file given ACIS-I chip
spec.find_stowed(res_dir)
# Extract spectra in the background region for each obsid
spec.bkg_spectrum(res_dir, multiobs, obsids)
# Define the annuli to be used for cluster spectrum extraction
N_ann = spec.find_spec_annuli(
    res_dir, Xdepro, Ydepro, bkg_area, z, R500, single_ann_spec, obsids
)
# Extract a spectrum in each annulus for each obsid
spec.extract_cl_spectra(res_dir, multiobs, obsids)
# Compute the ARF and RMF in each annulis of the SB profile
spec.arf_per_bin(res_dir, multiobs, obsids)
# Compute the hydrogen column density in the cluster direction
spec.hydrogen_column(res_dir)
# Simultaneously fit the cluster and background spectra
spec.fit_spec(res_dir, obsids, z)
# Plot the fit results
plt.plot_spectra(res_dir, obsids)
# Fit the temperature profile with a Vikhlinin model
spec.fit_T_prof(res_dir, R500, z)
# Plot the fit results
plt.plot_T_prof(res_dir, R500)
# Compute the conversion factors from surface brightness to emission measure
spec.XSB_to_EM_coef(res_dir, obsids, z)

print("------------------------------------------------------------")
print(colored("Estimation of ICM profiles", "cyan", None, ["bold"]))
print("------------------------------------------------------------")
# Initialize log-spaced integration radius maps
Rproj, r_map, los_step_map = icm.init_integ_maps(z, R500)
# Run the MCMC analysis to fit the surface brightness profile
icm.mcmc_ne(res_dir, Rproj, r_map, los_step_map, z, R500)
# Clean the chains
icm.clean_chains(res_dir, "ne")
# Find the best-fit density model
icm.best_ne_model(res_dir, Rproj, r_map, los_step_map, z, R500)
if N_ann > 2:
    # Run the MCMC analysis to fit the temperature profile
    icm.mcmc_pe(res_dir)
    # Clean the chains
    icm.clean_chains(res_dir, "pe")
# Find the best-fit ICM models
icm.best_icm_models(res_dir, z, R500, N_ann, Ysz)
# Compute the cooling luminosity if requested
if compute_Lcool:
    icm.cooling_lum(
        res_dir, z, tcool_th, Xdepro, Ydepro, multiobs, obsids, input_XSZ_file, do_err
    )

print("------------------------------------------------------------")
print(colored("Analysis figures", "cyan", None, ["bold"]))
print("------------------------------------------------------------")

# Plot the ICM profiles
plt.plot_icm_profiles(res_dir, file_ACCEPT, z)
plt.plot_2D_posteriors(res_dir, N_ann)
plt.adaptive_map(res_dir, z, R500)
plt.compute_Aphot(res_dir, z, R500)
plt.cluster_id_card(res_dir, source_name, z)

sp.call("cp param.py " + res_dir, shell=True)

te = time.time()
print(
    colored("=== *************************************** ===", "cyan", None, ["bold"])
)
print(colored("               End of program", "cyan", None, ["bold"]))
print(
    colored(
        "          Execution time: " + "{0:.1f}".format((te - ti) / 60.0) + " min",
        "cyan",
        None,
        ["bold"],
    )
)
print(
    colored("=== *************************************** ===", "cyan", None, ["bold"])
)
