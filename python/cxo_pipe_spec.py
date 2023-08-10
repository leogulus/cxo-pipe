import subprocess as sp
from termcolor import colored
from astropy.io import fits
import os
import logging
import numpy as np
import copy
import glob
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.coordinates import SkyCoord
from gdpyc import GasMap
import warnings
from tqdm import tqdm
import scipy.optimize as optimization
from sherpa.astro import ui

cosmo = FlatLambdaCDM(70.0, 0.3, Tcmb0=2.7255)


def find_stowed(res_dir):
    """
    Find stowed background associated with cluster and
    background spectra given the CCD_ID used for pointing

    Parameters
    __________
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Creates a stowed_files.txt file containing the paths to
    the stowed background files to be used

    """

    mer_dir = res_dir + "/results/"

    print(colored("Finding stowed background files...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "stowed_files.txt"):
        print(colored("Stowed background files already found", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        efile = mer_dir + "efile_repro_raw_clean_nopts.fits"
        hdu = fits.open(efile)
        header = hdu[1].header
        try:
            ccd_id = header["CCD_ID"]
        except:
            ccd_id = int(header["DSVAL2"][0])

        bkg_dir = os.environ["CXO_BKG_FILES"]
        pipe_dir = os.environ["CXO_PIPE"]

        if ccd_id == 0:
            stowed_cl = bkg_dir + "acis0D2005-09-01bgstow_ctiN0002.fits"
            reg_cl = pipe_dir + "tools/stowed_I0.reg"
            stowed_bkg = bkg_dir + "acis3D2005-09-01bgstow_ctiN0002.fits"
            reg_bkg = pipe_dir + "tools/stowed_I3.reg"
        elif ccd_id == 1:
            stowed_cl = bkg_dir + "acis1D2005-09-01bgstow_ctiN0002.fits"
            reg_cl = pipe_dir + "tools/stowed_I1.reg"
            stowed_bkg = bkg_dir + "acis2D2005-09-01bgstow_ctiN0002.fits"
            reg_bkg = pipe_dir + "tools/stowed_I2.reg"
        elif ccd_id == 2:
            stowed_cl = bkg_dir + "acis2D2005-09-01bgstow_ctiN0002.fits"
            reg_cl = pipe_dir + "tools/stowed_I2.reg"
            stowed_bkg = bkg_dir + "acis1D2005-09-01bgstow_ctiN0002.fits"
            reg_bkg = pipe_dir + "tools/stowed_I1.reg"
        elif ccd_id == 3:
            stowed_cl = bkg_dir + "acis3D2005-09-01bgstow_ctiN0002.fits"
            reg_cl = pipe_dir + "tools/stowed_I3.reg"
            stowed_bkg = bkg_dir + "acis0D2005-09-01bgstow_ctiN0002.fits"
            reg_bkg = pipe_dir + "tools/stowed_I0.reg"
        else:
            print(
                colored(
                    "Observations are not made with ACIS-I!\n"
                    + "--> This pipeline is not adapted",
                    "red",
                    None,
                    ["bold"],
                )
            )
            sys.exit()

        stowed_files = open(mer_dir + "stowed_files.txt", "w")
        stowed_files.write(stowed_cl + "\n")
        stowed_files.write(reg_cl + "\n")
        stowed_files.write(stowed_bkg + "\n")
        stowed_files.write(reg_bkg)
        stowed_files.close()


def bkg_spectrum(res_dir, multiobs, obsids):
    """
    Extracts the background spectrum. If there are multiple
    obsids of a single object: extracts the spectrum in
    each event file

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    multiobs: are there multiple obsids to consider? True/False
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates a *background* directory in the *results* folder of res_dir
    containing the background spectrum / spectra

    """

    mer_dir = res_dir + "/results/"
    bkg_dir = mer_dir + "background/"

    if not os.path.exists(bkg_dir):
        sp.call("mkdir " + bkg_dir, shell=True)

    print(colored("Extracting background spectra...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    specfiles = glob.glob(bkg_dir + "/*.pi")

    if len(specfiles) > 0:
        print(colored("Background spectra already extracted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        stowed_files = mer_dir + "stowed_files.txt"
        with open(stowed_files) as f:
            content = f.readlines()

        bkg_stowed_file = content[2][:-1]
        bkg_stowed_reg = content[3]

        tab_obsid = obsids.split(",")
        for obsid in tab_obsid:
            bkg_region_file = mer_dir + "bkg_region_" + obsid + ".reg"
            if multiobs:
                efile = mer_dir + "All_" + obsid + "_reproj_evt_nopts.fits"
            else:
                efile = mer_dir + "efile_repro_raw_clean_nopts.fits"
            out_file = bkg_dir + "bkg_spectrum_" + obsid
            sp.call(
                [
                    "bash",
                    "shell/extract_spectrum.sh",
                    efile,
                    bkg_region_file,
                    bkg_stowed_file,
                    bkg_stowed_reg,
                    out_file,
                ]
            )


def find_spec_annuli(
    res_dir, Xdepro, Ydepro, bkg_area, z, R500, single_ann_spec, obsids
):
    """
    Defines the annuli for the extraction of the cluster spectra
    based on the S/N in each annulus

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    Xdepro: the RA position of the deprojection center
    Ydepro: the Dec position of the deprojection center
    bkg_area: the area of the background region in pixel**2
    z: the cluster redshift
    R500: the cluster R500 radius in kpc
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates a *cluster* directory in the *results* folder of res_dir
    containing DS9 region files called spec_annulus_%i.reg to be used to
    extract the cluster spectra a different radii from the deprojection center.
    Returns the number of annuli

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"

    if not os.path.exists(cl_dir):
        sp.call("mkdir " + cl_dir, shell=True)

    print(
        colored(
            "Defining annuli for cluster spectrum extraction...", "blue", None, ["bold"]
        )
    )
    print("------------------------------------------------------------")

    if os.path.exists(cl_dir + "spec_annuli.reg"):
        print(
            colored("Cluster spectrum annuli already defined", "white", None, ["bold"])
        )
        print("------------------------------------------------------------")
        annuli_file = cl_dir + "spec_annuli.reg"
        with open(annuli_file) as f:
            content = f.readlines()
        N_ann = len(content) - 1
        return N_ann
    else:
        map_file = mer_dir + "wide_broad_thresh_nopts.img"
        hdu = fits.open(map_file)
        cl_header = hdu[0].header
        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        R500_pix = (
            ((R500 / d_a) * u.rad).to("arcsec")
            / (cl_header["CDELT2"] * 3600.0 / cl_header["CDELT2P"])
        ).value

        # Create map weighted by exposure / expo_max
        expo_file = mer_dir + "wide_broad_thresh.expmap"
        out_file = mer_dir + "w8_count_rate.img"
        hdu = fits.open(expo_file)
        expo_max = np.max(hdu[0].data)
        weight_expo = 1.0 / expo_max
        sp.call(
            [
                "bash",
                "shell/w8_count_rate_img.sh",
                map_file,
                expo_file,
                str(weight_expo),
                out_file,
            ]
        )

        # Estimate background count rate per unit area (in pixel**2)
        tab_obsid = obsids.split(",")
        reg_file = mer_dir + "bkg_region_" + tab_obsid[0] + ".reg"
        bkg_count_file = mer_dir + "bkg_counts.txt"
        sp.call(["bash", "shell/counts_in_reg.sh", out_file, reg_file, bkg_count_file])
        with open(bkg_count_file) as f:
            content = f.readlines()
        bkg_counts = float(content[5][9:-1])
        bkg_count_rate = bkg_counts / cl_header["exposure"] / bkg_area[0]

        if single_ann_spec:
            reg_file_name_i = cl_dir + "spec_annulus_1.reg"
            reg_file_i = open(reg_file_name_i, "w")
            reg_file_i.write("# Region file format: CIAO version 1.0\n")
            reg_file_i.write(
                "annulus("
                + str(Xdepro)
                + ","
                + str(Ydepro)
                + ","
                + str(0.15 * R500_pix)
                + ","
                + str(R500_pix)
                + ")"
            )
            reg_file_i.close()
            inner_rad_tab = [0.15 * R500_pix]
            outer_rad_tab = [R500_pix]
        else:
            # Loop to find annuli given S/N per annulus
            counts_file_name_i = mer_dir + "spec_annulus_counts_i.txt"
            inner_rad = 0.0
            outer_rad = 0.0
            index_ring = 1
            min_count = 400
            inner_rad_tab = []
            outer_rad_tab = []
            while inner_rad < R500_pix:
                N_tot = 0
                rad_add = 0.0
                S2N = 0.0
                while ((S2N < 30) | (N_tot < min_count)) & (rad_add < R500_pix):
                    if inner_rad < 20.0:
                        rad_add += 1.0
                    else:
                        rad_add += 20.0
                    outer_rad = inner_rad + rad_add
                    reg_file_name_i = (
                        cl_dir + "spec_annulus_" + str(index_ring) + ".reg"
                    )
                    reg_file_i = open(reg_file_name_i, "w")
                    reg_file_i.write("# Region file format: CIAO version 1.0\n")
                    reg_file_i.write(
                        "annulus("
                        + str(Xdepro)
                        + ","
                        + str(Ydepro)
                        + ","
                        + str(inner_rad)
                        + ","
                        + str(outer_rad)
                        + ")"
                    )
                    reg_file_i.close()
                    sp.call(
                        [
                            "bash",
                            "shell/counts_in_reg.sh",
                            out_file,
                            reg_file_name_i,
                            counts_file_name_i,
                        ]
                    )
                    with open(counts_file_name_i) as f:
                        content = f.readlines()
                    N_tot = float(content[5][9:-1])
                    area_ann_i = (np.pi * outer_rad ** 2) - (np.pi * inner_rad ** 2)
                    N_B = bkg_count_rate * cl_header["exposure"] * area_ann_i
                    S2N = (N_tot - N_B) / np.sqrt(N_tot)

                if rad_add < R500_pix:
                    inner_rad_tab.append(inner_rad)
                    outer_rad_tab.append(outer_rad)
                    index_ring += 1
                inner_rad += rad_add

            if index_ring == 1:
                reg_file_name_i = cl_dir + "spec_annulus_" + str(index_ring) + ".reg"
                reg_file_i = open(reg_file_name_i, "w")
                reg_file_i.write("# Region file format: CIAO version 1.0\n")
                reg_file_i.write(
                    "annulus("
                    + str(Xdepro)
                    + ","
                    + str(Ydepro)
                    + ","
                    + str(0.15 * R500_pix)
                    + ","
                    + str(R500_pix)
                    + ")"
                )
                reg_file_i.close()
                inner_rad_tab = [0.15 * R500_pix]
                outer_rad_tab = [R500_pix]

        reg_file_name = cl_dir + "spec_annuli.reg"
        reg_file = open(reg_file_name, "w")
        reg_file.write("# Region file format: CIAO version 1.0\n")
        for i in range(len(inner_rad_tab)):
            reg_file.write(
                "annulus("
                + str(Xdepro)
                + ","
                + str(Ydepro)
                + ","
                + str(inner_rad_tab[i])
                + ","
                + str(outer_rad_tab[i])
                + ")\n"
            )
        reg_file.close()

        return len(inner_rad_tab)


def extract_cl_spectra(res_dir, multiobs, obsids):
    """
    Extracts the cluster spectra in annuli. If there are multiple
    obsids of a single object: extracts spectra in each event file

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    multiobs: are there multiple obsids to consider? True/False
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates the spectrum files to be used by Sherpa in the *cluster*
    sub-directory of *results* in res_dir

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"

    print(colored("Extracting cluster spectra...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    specfiles = glob.glob(cl_dir + "/*.pi")

    if len(specfiles) > 0:
        print(colored("Cluster spectra already extracted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        stowed_files = mer_dir + "stowed_files.txt"
        with open(stowed_files) as f:
            content = f.readlines()
        bkg_stowed_file = content[0][:-1]
        bkg_stowed_reg = content[1][:-1]

        annuli_file = cl_dir + "spec_annuli.reg"
        with open(annuli_file) as f:
            content = f.readlines()
        N_ann = len(content) - 1

        tab_obsid = obsids.split(",")
        for obsid in tab_obsid:
            if multiobs:
                efile = mer_dir + "All_" + obsid + "_reproj_evt_nopts.fits"
            else:
                efile = mer_dir + "efile_repro_raw_clean_nopts.fits"

            for i in range(1, N_ann + 1):
                cl_region_file = cl_dir + "spec_annulus_" + str(i) + ".reg"
                out_file = cl_dir + "cl_spectrum_" + obsid + "_" + str(i)
                sp.call(
                    [
                        "bash",
                        "shell/extract_spectrum.sh",
                        efile,
                        cl_region_file,
                        bkg_stowed_file,
                        bkg_stowed_reg,
                        out_file,
                    ]
                )


def arf_per_bin(res_dir, multiobs, obsids):
    """
    Extracts the ARF in each bin considered for the surface
    brightness profile. To be used in the conversion to
    emission measure profile

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    multiobs: are there multiple obsids to consider? True/False
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates the ARF files to be used by Sherpa in the *ARF*
    sub-directory of *results* in res_dir

    """

    mer_dir = res_dir + "/results/"
    arf_dir = mer_dir + "ARF/"

    if not os.path.exists(arf_dir):
        sp.call("mkdir " + arf_dir, shell=True)

    print(colored("Extracting ARF per bin...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    arffiles = glob.glob(arf_dir + "/*.arf")

    if len(arffiles) > 0:
        print(colored("ARF already extracted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        sb_annuli_file = mer_dir + "SB_annuli.reg"
        with open(sb_annuli_file) as f:
            content = f.readlines()

        tab_obsid = obsids.split(",")
        for obsid in tab_obsid:
            if multiobs:
                efile = mer_dir + "All_" + obsid + "_reproj_evt_nopts.fits"
            else:
                efile = mer_dir + "efile_repro_raw_clean_nopts.fits"

            for i in range(1, len(content)):
                reg_annulus = content[i][:-1]
                out_file = arf_dir + "cl_spectrum_" + obsid + "_" + str(i)
                sp.call(
                    [
                        "bash",
                        "shell/extract_spectrum_nobkg.sh",
                        efile,
                        reg_annulus,
                        out_file,
                    ]
                )

            noARF = []
            withARF = []
            for i in range(1, len(content)):
                if not (
                    os.path.exists(
                        arf_dir + "/cl_spectrum_" + obsid + "_" + str(i) + ".pi"
                    )
                    & os.path.exists(
                        arf_dir + "/cl_spectrum_" + obsid + "_" + str(i) + ".arf"
                    )
                    & os.path.exists(
                        arf_dir + "/cl_spectrum_" + obsid + "_" + str(i) + ".rmf"
                    )
                    & os.path.exists(
                        arf_dir + "/cl_spectrum_" + obsid + "_" + str(i) + "_grp.pi"
                    )
                ):
                    noARF.append(i)
                else:
                    withARF.append(i)

            for ind_no_arf in noARF:
                not_done = True
                for ind_with_arf in withARF:
                    if (ind_no_arf < ind_with_arf) & not_done:
                        sp.call(
                            "cp "
                            + arf_dir
                            + "/cl_spectrum_"
                            + obsid
                            + "_"
                            + str(ind_with_arf)
                            + ".pi"
                            + " "
                            + arf_dir
                            + "/cl_spectrum_"
                            + obsid
                            + "_"
                            + str(ind_no_arf)
                            + ".pi",
                            shell=True,
                        )
                        sp.call(
                            "cp "
                            + arf_dir
                            + "/cl_spectrum_"
                            + obsid
                            + "_"
                            + str(ind_with_arf)
                            + ".arf"
                            + " "
                            + arf_dir
                            + "/cl_spectrum_"
                            + obsid
                            + "_"
                            + str(ind_no_arf)
                            + ".arf",
                            shell=True,
                        )
                        sp.call(
                            "cp "
                            + arf_dir
                            + "/cl_spectrum_"
                            + obsid
                            + "_"
                            + str(ind_with_arf)
                            + ".rmf"
                            + " "
                            + arf_dir
                            + "/cl_spectrum_"
                            + obsid
                            + "_"
                            + str(ind_no_arf)
                            + ".rmf",
                            shell=True,
                        )
                        sp.call(
                            "cp "
                            + arf_dir
                            + "/cl_spectrum_"
                            + obsid
                            + "_"
                            + str(ind_with_arf)
                            + "_grp.pi"
                            + " "
                            + arf_dir
                            + "/cl_spectrum_"
                            + obsid
                            + "_"
                            + str(ind_no_arf)
                            + "_grp.pi",
                            shell=True,
                        )
                        not_done = False


def hydrogen_column(res_dir):
    """
    Use results from Kalberla et al. (2005) to fix the
    neutral hydrogen column density along the line of sight

    Parameters
    __________
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Creates a .npy file in the *results* directory in res_dir
    containing the value of n_H x 10^-22

    """

    mer_dir = res_dir + "/results/"

    print(colored("Computing hydrogen column density...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    nH_file = mer_dir + "nH_value.npy"

    if os.path.exists(nH_file):
        print(
            colored("Hydrogen column density already computed", "white", None, ["bold"])
        )
        print("------------------------------------------------------------")
    else:
        map_file = mer_dir + "wide_broad_thresh_nopts.img"
        hdu = fits.open(map_file)
        cent_RA = hdu[0].header["RA_NOM"]
        cent_Dec = hdu[0].header["DEC_NOM"]

        coords = SkyCoord(
            ra=cent_RA * u.deg,
            dec=cent_Dec * u.deg,
            unit=(u.deg, u.deg),
            frame="fk5",
            obstime="J2000.00",
        )

        nH_val = GasMap.nh(coords, nhmap="DL").value * 1e-22
        np.save(nH_file, nH_val)


def fit_spec(res_dir, obsids, z):
    """
    Simultaneously fit background and source spectra in each
    annulus defined with find_spec_annuli using Sherpa

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    obsids: the list of obsids given as a comma-separated string of numbers
    z: the cluster redshift

    Returns
    _______
    Creates .npz files in the *cluster* folder of the *results*
    directory in res_dir containing the spectroscopic temperature
    profile and the spectrum fits in each annulus

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    bkg_dir = mer_dir + "background/"

    print(colored("Fitting spectra...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_T_prof = cl_dir + "spectro_T_prof.npz"

    if os.path.exists(file_T_prof):
        print(colored("Spectra already fitted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        efile = mer_dir + "efile_repro_raw_clean_nopts.fits"
        reg_file_cl = cl_dir + "spec_annuli.reg"
        efile_in = efile + "[bin sky=@" + reg_file_cl + "]"
        file_out = mer_dir + "spec_ann_stat.fits"
        sp.call(["bash", "shell/extract_content.sh", efile_in, file_out])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdu = fits.open(file_out)
            tab_area_cl = hdu[1].data["AREA"]

        tab_obsid = obsids.split(",")

        logger = logging.getLogger("sherpa")
        logger.setLevel(logging.WARN)

        N_spec_reg = tab_area_cl.size
        ICM_T_tab = np.zeros(N_spec_reg)
        ICM_T_tab_err = np.zeros(N_spec_reg)
        tab_area_bkg = [0]

        for ind_ann in tqdm(range(1, N_spec_reg + 1)):

            buff_T_err = 1.0
            buff_T = 1.0
            SNR_bin = 3
            T_max = 100.0

            while (SNR_bin < 10) & (buff_T_err / buff_T == 1.0) | (T_max > 30.0):

                ui.clean()
                fit_ind = 1
                for obsid in tab_obsid:
                    file_area_bkg = mer_dir + "bkg_area_" + obsid + ".txt"
                    with open(file_area_bkg) as f:
                        content = f.readlines()
                    area_bkg = float(content[0])
                    tab_area_bkg.append(area_bkg)
                    ui.load_data(
                        fit_ind,
                        cl_dir + "cl_spectrum_" + obsid + "_" + str(ind_ann) + ".pi",
                    )
                    fit_ind += 1
                    tab_area_bkg.append(area_bkg)
                    ui.load_data(fit_ind, bkg_dir + "bkg_spectrum_" + obsid + ".pi")
                    fit_ind += 1

                for i in range(1, fit_ind):
                    full_spec = ui.get_data_plot(i)
                    particle_spec = ui.get_bkg(i)
                    wscale = np.where((full_spec.x > 9) & (full_spec.x < 12))
                    newscale = np.trapz(
                        ui.get_counts(i)[wscale], full_spec.x[wscale]
                    ) / np.trapz(particle_spec.counts[wscale], full_spec.x[wscale])
                    if newscale != 0:
                        area_scale = ui.get_backscal(i)
                        bkg_scale = ui.get_bkg_scale(i)
                        ui.set_backscal(i, area_scale * newscale / bkg_scale)
                    ui.subtract(i)
                    ui.group_snr(i, SNR_bin)

                if (T_max != 100.0) & (T_max > 30.0):
                    ui.notice(0.7, 7)
                else:
                    ui.notice(0.7, 12)

                ui.set_method("neldermead")
                ui.set_stat("chi2gehrels")

                nH_val = float(np.load(mer_dir + "nH_value.npy"))

                for i in range(1, fit_ind):
                    ui.xsphabs.nH.nH = nH_val
                    ui.freeze(ui.xsphabs.nH.nH)
                    ui.xsapec.kt.redshift = z
                    ui.freeze(ui.xsapec.kt.redshift)
                    ui.xsapec.kt.Abundanc = 0.3
                    ui.freeze(ui.xsapec.kt.Abundanc)
                    ui.xsapec.kt.kt = 12.0
                    ui.xsapec.kt.norm = 7e-4

                    ui.powlaw1d.p1.ref = 10
                    ui.powlaw1d.p1.gamma.min = -100
                    ui.powlaw1d.p1.gamma.max = 100
                    ui.powlaw1d.p1.gamma = 2
                    ui.powlaw1d.p1.ampl = 1e-2
                    ui.xsapec.ktb.redshift = 0.0
                    ui.freeze(ui.xsapec.ktb.redshift)
                    ui.xsapec.ktb.Abundanc = 1.0
                    ui.freeze(ui.xsapec.ktb.Abundanc)
                    ui.xsapec.ktb.kt = 0.18
                    ui.freeze(ui.xsapec.ktb.kt)
                    ui.xsapec.ktb.norm = 5e-2

                    tab_area_fact = tab_area_cl / tab_area_bkg[i]

                    if i % 2:
                        ui.set_source(
                            i,
                            ui.xsphabs.nH * ui.xsapec.kt
                            + tab_area_fact[ind_ann - 1]
                            * (
                                ui.const1d.a * ui.xsapec.ktb
                                + ui.const1d.b * ui.powlaw1d.p1
                            ),
                        )
                    else:
                        ui.set_source(
                            i,
                            ui.const1d.a * ui.xsapec.ktb
                            + ui.const1d.b * ui.powlaw1d.p1,
                        )

                ui.fit()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if N_spec_reg == 1:
                        ui.covar(kt.kt)
                    else:
                        ui.covar(kt.kt)

                i = 1
                for obsid in tab_obsid:
                    tab_area_fact = tab_area_cl / tab_area_bkg[i]
                    spec_fit = copy.deepcopy(ui.get_fit_plot(i))
                    bkg_fit = copy.deepcopy(ui.get_fit_plot(i + 1))
                    save_spec = (
                        cl_dir + "spec_fit_" + obsid + "_" + str(ind_ann) + ".npz"
                    )
                    np.savez(
                        save_spec,
                        datax=spec_fit.dataplot.x,
                        datay=spec_fit.dataplot.y,
                        dataxerr=spec_fit.dataplot.xerr,
                        datayerr=spec_fit.dataplot.yerr,
                        fitx=spec_fit.modelplot.x,
                        fity=spec_fit.modelplot.y,
                        bkgdatx=bkg_fit.dataplot.x,
                        bkgdaty=bkg_fit.dataplot.y,
                        bkgdatxerr=bkg_fit.dataplot.xerr,
                        bkgdatyerr=bkg_fit.dataplot.yerr,
                        bkgfitx=bkg_fit.modelplot.x,
                        bkgfity=bkg_fit.modelplot.y,
                        bkgsc=tab_area_fact[ind_ann - 1],
                    )
                    i += 2

                if N_spec_reg == 1:
                    final_res = ui.get_covar_results()
                else:
                    final_res = ui.get_covar_results()

                ICM_T_tab[ind_ann - 1] = final_res.parvals[0]
                buff_T = final_res.parvals[0]
                T_max = final_res.parvals[0]

                if (final_res.parmins[0] is None) & (final_res.parmaxes[0] is None):
                    ICM_T_tab_err[ind_ann - 1] = final_res.parvals[0]
                    buff_T_err = final_res.parvals[0]
                elif (final_res.parmins[0] is not None) & (
                    final_res.parmaxes[0] is None
                ):
                    ICM_T_tab_err[ind_ann - 1] = np.abs(final_res.parmins[0])
                    buff_T_err = np.abs(final_res.parmins[0])
                else:
                    ICM_T_tab_err[ind_ann - 1] = final_res.parmaxes[0]
                    buff_T_err = final_res.parmaxes[0]

                if buff_T_err / buff_T < 0.05:
                    ICM_T_tab_err[ind_ann - 1] = 0.15 * buff_T
                    buff_T_err = 0.15 * buff_T

                SNR_bin += 1

        np.savez(file_T_prof, T=ICM_T_tab, Terr=ICM_T_tab_err)


def V06_model(args, x_tab, T_tab, T_tab_err, return_chi):

    T_model = (
        args[0]
        * (((x_tab / 0.045) ** 1.9 + args[1]) / ((x_tab / 0.045) ** 1.9 + 1.0))
        * (1.0 + (x_tab / 0.6) ** 2) ** (-0.45)
    )

    if return_chi:
        x_tab2 = np.linspace(0, 1, 100)
        T_model2 = (
            args[0]
            * (((x_tab2 / 0.045) ** 1.9 + args[1]) / ((x_tab2 / 0.045) ** 1.9 + 1.0))
            * (1.0 + (x_tab2 / 0.6) ** 2) ** (-0.45)
        )
        speed = np.diff(T_model2)
        acc = np.diff(speed)
        if np.mean(acc) > 0:
            return 1e10 * np.ones(T_tab.size)
        elif not all(Te > 0.1 for Te in T_model2):
            return 1e10 * np.ones(T_tab.size)
        else:
            return (T_tab - T_model) / T_tab_err
    else:
        return T_model


def fit_T_prof(res_dir, R500, z):
    """
    Fit the spectroscopic temperature profile with the
    parametric model from Vikhlinin+2006

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    R500: the cluster R500 radius in kpc
    z: the cluster redshift

    Returns
    _______
    Creates a .npz file in the *cluster* folder of the *results*
    directory in res_dir containing the model of the spectroscopic
    temperature profile to be used to compute the emission measure

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"

    print(colored("Fitting temperature profile...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    save_fit = cl_dir + "T_prof_fit.npz"

    if os.path.exists(save_fit):
        print(colored("Temperature profile already fitted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        map_file = mer_dir + "wide_broad_thresh_nopts.img"
        hdu = fits.open(map_file)
        header = hdu[0].header
        pix2rad = ((header["CDELT2"] / header["CDELT2P"]) * u.deg).to("rad").value
        d_a = cosmo.angular_diameter_distance(z).to("kpc").value

        file_T_prof = cl_dir + "spectro_T_prof.npz"
        data_T_prof = np.load(file_T_prof)
        T_prof = data_T_prof["T"]
        T_prof_err = data_T_prof["Terr"]

        std_T = np.std(T_prof)
        for i in range(T_prof_err.size):
            T_err_i = T_prof_err[i]
            if T_err_i < std_T / 5.0:
                T_prof_err[i] = std_T

        file_ann = cl_dir + "spec_annuli.reg"
        with open(file_ann) as f:
            content = f.readlines()

        if len(content) < 3:
            rad_ann_mid = np.array([(0.15 * R500 + R500) / 2.0])
            rad_ann_err = np.array([R500]) - rad_ann_mid
        else:
            rad_ann = [0.0]
            for i in range(1, len(content)):
                rad_ann.append(float(content[i].split(",")[-1][:-2]) * pix2rad * d_a)

            rad_ann = np.array(rad_ann)
            rad_ann_mid = ((rad_ann + np.roll(rad_ann, -1)) / 2.0)[:-1]
            rad_ann_mid_R500 = rad_ann_mid / R500
            rad_ann_err = rad_ann_mid - rad_ann[:-1]

        mean_ICM_T = np.mean(T_prof)
        std_ICM_T = np.mean(T_prof_err)
        rad_plot = np.logspace(0, np.log10(3.0 * R500), 1000)

        if rad_ann_mid.size > 2:
            N_MC = 200
            T_model_tab = np.zeros((N_MC, rad_plot.size))
            param_T = [1.35 * mean_ICM_T, 0.35]
            for i in tqdm(range(N_MC)):
                popt = optimization.leastsq(
                    V06_model,
                    param_T,
                    args=(
                        rad_ann_mid_R500,
                        np.random.normal(T_prof, T_prof_err),
                        T_prof_err,
                        1,
                    ),
                    maxfev=int(1e6),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted_param = popt[0]
                T_model = V06_model(
                    fitted_param, rad_plot / R500, T_prof, T_prof_err, 0
                )
                T_model_tab[i, :] = T_model

            T_model_bf = np.mean(T_model_tab, axis=0)
            T_model_bf_std = np.std(T_model_tab, axis=0)
        else:
            x_tab = rad_plot / R500
            T_model_bf = (
                mean_ICM_T
                * 1.35
                * (((x_tab / 0.045) ** 1.9 + 0.45) / ((x_tab / 0.045) ** 1.9 + 1.0))
                * (1.0 + (x_tab / 0.6) ** 2) ** (-0.45)
            )
            T_model_bf_std = (
                std_ICM_T
                * 1.35
                * (((x_tab / 0.045) ** 1.9 + 0.45) / ((x_tab / 0.045) ** 1.9 + 1.0))
                * (1.0 + (x_tab / 0.6) ** 2) ** (-0.45)
            )

        np.savez(
            save_fit,
            datax=rad_ann_mid,
            datay=T_prof,
            dataxerr=rad_ann_err,
            datayerr=T_prof_err,
            fitx=rad_plot,
            fity=T_model_bf,
            fityerr=T_model_bf_std,
        )


def XSB_to_EM_coef(res_dir, obsids, z):
    """
    Compute the conversion factors from surface brightness
    to emission measure using the temperature model
    estimated with fit_T_prof

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    obsids: the list of obsids given as a comma-separated string of numbers
    z: the cluster redshift

    Returns
    _______
    Creates a .npy file in the *cluster* folder of the *results*
    directory in res_dir containing the distribution of the conversion factor
    from XSB to EM obtained by sampling the temperature model uncertainties
    in each bin of the surface brightness profile

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    arf_dir = mer_dir + "ARF/"

    print(colored("Computing XSB --> EM conversion factors...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_conv_tab = cl_dir + "conv_tab.npy"

    if os.path.exists(file_conv_tab):
        print(colored("Conversion factors already computed", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        xsb_file = mer_dir + "XSB_profile_rmid.fits"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdu = fits.open(xsb_file)
            header = hdu[1].header
            xsb_prof = hdu[1].data

        N_counts_perbin = xsb_prof["net_counts"]
        Rmid = xsb_prof["rmid"] * header["TCDLT6"] / 60.0
        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        Rmid_kpc = d_a * (Rmid * u.arcmin).to("radian").value
        expo_tot = header["exposure"]

        area_perbin = xsb_prof["area"] * header["TCDLT11"] / 3600.0
        area_perbin_kpc = (area_perbin * u.arcmin ** 2).to(u.rad ** 2).value * d_a ** 2

        T_file = cl_dir + "T_prof_fit.npz"
        T_data = np.load(T_file)
        T_rad = T_data["fitx"]
        T_prof = T_data["fity"]
        T_prof_err = T_data["fityerr"]

        T_prof_int = np.interp(Rmid_kpc, T_rad, T_prof)
        T_prof_int_err = np.interp(Rmid_kpc, T_rad, T_prof_err)

        tab_obsid = obsids.split(",")

        logger = logging.getLogger("sherpa")
        logger.setLevel(logging.WARN)
        ui.clean()

        nH_val = float(np.load(mer_dir + "nH_value.npy"))
        ui.xsphabs.nH.nH = nH_val
        ui.freeze(ui.xsphabs.nH.nH)
        ui.xsapec.kt.redshift = z
        ui.freeze(ui.xsapec.kt.redshift)
        ui.xsapec.kt.Abundanc = 0.3
        ui.freeze(ui.xsapec.kt.Abundanc)
        ui.xsapec.kt.kt = 12.0
        ui.xsapec.kt.norm = 7e-4
        ui.set_source(ui.xsapec.kt * ui.xsphabs.nH)

        N_MC = 100
        conv_tab_obs = np.zeros((N_counts_perbin.size, N_MC, len(tab_obsid)))
        expo_obs = np.zeros(len(tab_obsid))

        for i in tqdm(range(N_counts_perbin.size)):
            for ind, obsid in enumerate(tab_obsid):
                spec_file = arf_dir + "cl_spectrum_" + obsid + "_" + str(i + 1) + ".pi"
                ui.load_data(spec_file)
                for nmc in range(N_MC):
                    T_MC = T_prof_int[i] + np.random.normal() * T_prof_int_err[i]
                    if T_MC < 0.5:
                        ui.xsapec.kt.kt = 0.5
                    elif T_MC > 30.0:
                        ui.xsapec.kt.kt = 30.0
                    else:
                        ui.xsapec.kt.kt = T_MC

                    model_fit = ui.get_fit_plot()

                    # Find energy band [0.7,2] keV used for X-ray surface brightness profile estimate
                    wXSB_prof = np.where(
                        (model_fit.modelplot.x > 0.7) & (model_fit.modelplot.x < 2.0)
                    )
                    # Compute total count in the [0.7,2] keV band for the current normalization
                    count_mod = (
                        np.trapz(
                            model_fit.modelplot.y[wXSB_prof],
                            model_fit.modelplot.x[wXSB_prof],
                        )
                        * ui.get_exposure()
                    )
                    expo_obs[ind] = ui.get_exposure()
                    # Rescale the normalization knowing the real total count in the spectrum extraction region
                    N_counts_obs = (N_counts_perbin[i] / expo_tot) * ui.get_exposure()
                    norm_val = ui.xsapec.kt.norm.val * N_counts_obs / count_mod
                    EM_val = (
                        norm_val
                        / (
                            area_perbin_kpc[i]
                            * 1e-14
                            / (4.0 * np.pi * (d_a * u.kpc * (1 + z)) ** 2)
                        ).value
                    )
                    CR_val = N_counts_obs / area_perbin[i] / ui.get_exposure()
                    conv_tab_obs[i, nmc, ind] = EM_val / CR_val

        num_sum = 0
        den_sum = 0
        for i in range(len(tab_obsid)):
            num_sum += expo_obs[i] * conv_tab_obs[:, :, i]
            den_sum += expo_obs[i]

        conv_tab = num_sum / den_sum
        np.save(file_conv_tab, conv_tab)

        # In each bin compute the normalization coefficient for 0.1 < Te < 30.1 keV
        norm_tab = np.zeros((N_counts_perbin.size, 31))
        Te_tab = np.linspace(0.1, 30.1, 31)
        for i in tqdm(range(N_counts_perbin.size)):
            spec_file = (
                arf_dir + "cl_spectrum_" + tab_obsid[0] + "_" + str(i + 1) + ".pi"
            )
            ui.load_data(spec_file)
            for j in range(Te_tab.size):
                ui.xsapec.kt.kt = Te_tab[j]
                model_fit = ui.get_fit_plot()
                # Find energy band [0.7,2] keV used for X-ray surface brightness profile estimate
                wXSB_prof = np.where(
                    (model_fit.modelplot.x > 0.7) & (model_fit.modelplot.x < 2.0)
                )
                # Compute total count in the [0.7,2] keV band for the current normalization
                count_mod = (
                    np.trapz(
                        model_fit.modelplot.y[wXSB_prof],
                        model_fit.modelplot.x[wXSB_prof],
                    )
                    * ui.get_exposure()
                )
                # Rescale the normalization knowing the real total count in the spectrum extraction region
                N_counts_obs = (N_counts_perbin[i] / expo_tot) * ui.get_exposure()
                norm_val = ui.xsapec.kt.norm.val * N_counts_obs / count_mod
                EM_val = (
                    norm_val
                    / (
                        area_perbin_kpc[i]
                        * 1e-14
                        / (4.0 * np.pi * (d_a * u.kpc * (1 + z)) ** 2)
                    ).value
                )
                CR_val = N_counts_obs / area_perbin[i] / ui.get_exposure()
                norm_tab[i, j] = EM_val / CR_val

        file_save_res = cl_dir + "Tab_conv_results.fits"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.writeto(file_save_res, Te_tab, overwrite=True)
            fits.append(file_save_res, norm_tab, overwrite=True)
