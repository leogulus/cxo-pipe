import subprocess as sp
from termcolor import colored
import glob
import sys
import os
import time
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Tophat2DKernel
import warnings
from astropy import wcs
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.table import Table
from astropy.io import ascii

cosmo = FlatLambdaCDM(70.0, 0.3, Tcmb0=2.7255)


def import_data(obsids, res_dir):
    """
    Download data from the Chandra archive based on list of obsids

    Parameters
    __________
    obsids: the list of obsids given as a comma-separated string of numbers
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Folders in the current working directory containing the data,
    using the obsids as forlder names

    """

    if not os.path.exists(res_dir):
        sp.call("mkdir " + res_dir, shell=True)

    tab_obsid = obsids.split(",")
    print(colored("Importing data...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    for obsid in tab_obsid:
        import_dir = res_dir + obsid
        if os.path.exists(import_dir):
            print(
                colored(
                    "Data from obsid " + obsid + " already downloaded",
                    "white",
                    None,
                    ["bold"],
                )
            )
        else:
            sp.call(["bash", "shell/import_data.sh", obsid])


def reprocess_data(obsids, res_dir):
    """
    Reprocess the evt1 event files

    Parameters
    __________
    obsids: the list of obsids given as a comma-separated string of numbers
    res_dir: the result directory named after the cluster name

    Returns
    _______
    New directories in res_dir named after the different obsids,
    containing the reprocessed event files

    """

    tab_obsid = obsids.split(",")
    print("------------------------------------------------------------")
    print(colored("Reprocessing data...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    for obsid in tab_obsid:
        repro_dir = res_dir + obsid
        try:
            rawfile = glob.glob(repro_dir + "/*repro_evt2.fits")[0]
            print(
                colored(
                    "Data from obsid " + obsid + " already reprocessed",
                    "white",
                    None,
                    ["bold"],
                )
            )
            print("------------------------------------------------------------")
        except IndexError:
            sp.call(["bash", "shell/reprocess_data.sh", obsid, repro_dir])
            sp.call("rm -rf " + obsid, shell=True)


def energy_cut(obsids, res_dir):
    """
    Create three event files by keeping only the energy ranges
    3-5 keV, 0.7-7 keV, and 0.7-2 keV

    Parameters
    __________
    obsids: the list of obsids given as a comma-separated string of numbers
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Three event files in res_dir with restricted energy ranges

    """
    tab_obsid = obsids.split(",")

    print(colored("Cutting energy range...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    for obsid in tab_obsid:
        repro_dir = res_dir + obsid
        if os.path.exists(repro_dir + "/efile_repro_0.7_7.0.fits"):
            print(
                colored(
                    "Energy cut already applied for obsid " + obsid,
                    "white",
                    None,
                    ["bold"],
                )
            )
            print("------------------------------------------------------------")
        else:
            try:
                rawfile = glob.glob(repro_dir + "/*repro_evt2.fits")[0]
            except IndexError:
                print(
                    colored(
                        "No reprocessed event file in "
                        + obsid
                        + " folder!\n"
                        + "--> Reprocess data first",
                        "red",
                        None,
                        ["bold"],
                    )
                )
                sys.exit()

            infile_cut_3_5 = rawfile + "[ccd_id=0:3][energy=3000:5000]"
            infile_cut_07_7 = rawfile + "[ccd_id=0:3][energy=700:7000]"
            infile_cut_07_2 = rawfile + "[ccd_id=0:3][energy=700:2000]"

            outfile_cut_3_5 = repro_dir + "/efile_repro_3.0_5.0.fits"
            outfile_cut_07_7 = repro_dir + "/efile_repro_0.7_7.0.fits"
            outfile_cut_07_2 = repro_dir + "/efile_repro_0.7_2.0.fits"

            sp.call(["bash", "shell/efile_cut.sh", infile_cut_3_5, outfile_cut_3_5])
            sp.call(["bash", "shell/efile_cut.sh", infile_cut_07_7, outfile_cut_07_7])
            sp.call(["bash", "shell/efile_cut.sh", infile_cut_07_2, outfile_cut_07_2])


def remove_flares(obsids, res_dir):
    """
    Remove flares from the raw event file and to the ones
    restricted to the energy ranges 3-5 keV, 0.7-7 keV, and 0.7-2 keV

    Parameters
    __________
    obsids: the list of obsids given as a comma-separated string of numbers
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Four event files (*_clean.fits) in the sub-directories of res_dir
    without contaminating events from flares

    """

    tab_obsid = obsids.split(",")

    print(colored("Removing flares...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    for obsid in tab_obsid:
        repro_dir = res_dir + obsid

        if not os.path.exists(repro_dir + "/efile_repro_0.7_7.0.fits"):
            print(
                colored(
                    "No event file with restricted energy range in "
                    + obsid
                    + " folder!\n"
                    + "--> Apply energy cut first",
                    "red",
                    None,
                    ["bold"],
                )
            )
            sys.exit()

        if os.path.exists(repro_dir + "/efile_repro_0.7_7.0_clean.fits"):
            print(
                colored(
                    "Flares already removed for obsid " + obsid, "white", None, ["bold"]
                )
            )
            print("------------------------------------------------------------")
        else:
            rawfile = glob.glob(repro_dir + "/*repro_evt2.fits")[0]
            infile_cut_3_5 = repro_dir + "/efile_repro_3.0_5.0.fits"
            infile_cut_07_7 = repro_dir + "/efile_repro_0.7_7.0.fits"
            infile_cut_07_2 = repro_dir + "/efile_repro_0.7_2.0.fits"

            lc_raw = rawfile[:-5] + ".lc"
            lc_3_5 = infile_cut_3_5[:-5] + ".lc"
            lc_07_7 = infile_cut_07_7[:-5] + ".lc"
            lc_07_2 = infile_cut_07_2[:-5] + ".lc"

            gti_raw = rawfile[:-5] + ".gti"
            gti_3_5 = infile_cut_3_5[:-5] + ".gti"
            gti_07_7 = infile_cut_07_7[:-5] + ".gti"
            gti_07_2 = infile_cut_07_2[:-5] + ".gti"

            out_raw = repro_dir + "/efile_repro_raw_clean.fits"
            outfile_cut_3_5 = repro_dir + "/efile_repro_3.0_5.0_clean.fits"
            outfile_cut_07_7 = repro_dir + "/efile_repro_0.7_7.0_clean.fits"
            outfile_cut_07_2 = repro_dir + "/efile_repro_0.7_2.0_clean.fits"

            sp.call(
                ["bash", "shell/remove_flares.sh", rawfile, lc_raw, gti_raw, out_raw]
            )
            sp.call(
                [
                    "bash",
                    "shell/remove_flares.sh",
                    infile_cut_3_5,
                    lc_3_5,
                    gti_3_5,
                    outfile_cut_3_5,
                ]
            )
            sp.call(
                [
                    "bash",
                    "shell/remove_flares.sh",
                    infile_cut_07_7,
                    lc_07_7,
                    gti_07_7,
                    outfile_cut_07_7,
                ]
            )
            sp.call(
                [
                    "bash",
                    "shell/remove_flares.sh",
                    infile_cut_07_2,
                    lc_07_2,
                    gti_07_2,
                    outfile_cut_07_2,
                ]
            )


def reproj_obsids(obsids, res_dir):
    """
    Reproject event files from different obsids on a common grid
    Use the raw event file and the ones restricted to 3-5 keV, 0.7-7 keV, and 0.7-2 keV

    Parameters
    __________
    obsids: the list of obsids given as a comma-separated string of numbers
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Creates a *results* folder in res_dir containing the merged event files
    and the images of the cluster in the 3-5 keV, 0.7-7 keV, and 0.7-2 keV bands

    """

    mer_dir = res_dir + "/results/"
    if not os.path.exists(mer_dir):
        sp.call("mkdir " + mer_dir, shell=True)

    print(colored("Reprojecting event files...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "/efile_repro_raw_clean.fits"):
        print(colored("Event files already reprojected", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        tab_obsid = obsids.split(",")
        list_evt_files_raw = ""
        list_evt_files_3_5 = ""
        list_evt_files_07_7 = ""
        list_evt_files_07_2 = ""

        for obsid in tab_obsid:
            repro_dir = res_dir + obsid
            list_evt_files_raw += "," + repro_dir + "/efile_repro_raw_clean.fits"
            list_evt_files_3_5 += "," + repro_dir + "/efile_repro_3.0_5.0_clean.fits"
            list_evt_files_07_7 += "," + repro_dir + "/efile_repro_0.7_7.0_clean.fits"
            list_evt_files_07_2 += "," + repro_dir + "/efile_repro_0.7_2.0_clean.fits"

        sp.call(
            [
                "bash",
                "shell/reproj_obsids.sh",
                list_evt_files_raw[1:],
                mer_dir + "/All_",
            ]
        )
        sp.call(
            [
                "bash",
                "shell/merge_obsids.sh",
                list_evt_files_3_5[1:],
                mer_dir + "/HB_",
                "3:5:4",
            ]
        )
        sp.call(
            [
                "bash",
                "shell/merge_obsids.sh",
                list_evt_files_07_7[1:],
                mer_dir + "/wide_",
                "broad",
            ]
        )
        sp.call(
            [
                "bash",
                "shell/merge_obsids.sh",
                list_evt_files_07_2[1:],
                mer_dir + "/SB_",
                "0.7:2:1",
            ]
        )
        sp.call(
            "mv "
            + mer_dir
            + "/All_merged_evt.fits "
            + mer_dir
            + "/efile_repro_raw_clean.fits",
            shell=True,
        )


def make_images(obsids, res_dir):
    """
    Making images of the cluster using the event files
    restricted to the 3-5 keV, 0.7-7 keV, and 0.7-2 keV bands

    Parameters
    __________
    obsids: the list of obsids given as a comma-separated string of numbers
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Creates a *results* folder in res_dir containing the cleaned event files
    and the images of the cluster in the 3-5 keV, 0.7-7 keV, and 0.7-2 keV bands

    """

    mer_dir = res_dir + "/results/"
    if not os.path.exists(mer_dir):
        sp.call("mkdir " + mer_dir, shell=True)

    print(colored("Making cluster maps...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "/wide_broad_thresh.img"):
        print(colored("Maps already made", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        repro_dir = res_dir + obsids
        sp.call(
            "cp " + repro_dir + "/efile_repro_raw_clean.fits " + mer_dir, shell=True
        )
        sp.call(
            "cp " + repro_dir + "/efile_repro_3.0_5.0_clean.fits " + mer_dir, shell=True
        )
        sp.call(
            "cp " + repro_dir + "/efile_repro_0.7_7.0_clean.fits " + mer_dir, shell=True
        )
        sp.call(
            "cp " + repro_dir + "/efile_repro_0.7_2.0_clean.fits " + mer_dir, shell=True
        )

        asol1_file = glob.glob(repro_dir + "/*_asol1.fits")[0]
        sp.call("cp " + asol1_file + " " + mer_dir, shell=True)

        bpix1_file = glob.glob(repro_dir + "/*repro_bpix1.fits")[0]
        sp.call("cp " + bpix1_file + " " + mer_dir, shell=True)

        msk1_file = glob.glob(repro_dir + "/*_msk1.fits")[0]
        sp.call("cp " + msk1_file + " " + mer_dir, shell=True)

        infile_3_5 = mer_dir + "/efile_repro_3.0_5.0_clean.fits"
        infile_07_7 = mer_dir + "/efile_repro_0.7_7.0_clean.fits"
        infile_07_2 = mer_dir + "/efile_repro_0.7_2.0_clean.fits"

        sp.call(["bash", "shell/make_images.sh", infile_3_5, mer_dir + "/HB_", "3:5:4"])
        sp.call(
            ["bash", "shell/make_images.sh", infile_07_7, mer_dir + "/wide_", "broad"]
        )
        sp.call(
            ["bash", "shell/make_images.sh", infile_07_2, mer_dir + "/SB_", "0.7:2:1"]
        )


def make_psf_map(res_dir):
    """
    Making the psf map associated with the image in the 0.7-7 keV band

    Parameters
    __________
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Creates the psf map of the 0.7-7 keV map in the *results* folder within res_dir

    """

    mer_dir = res_dir + "/results/"

    print(colored("Making PSF map...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "/psfmap_wide.fits"):
        print(colored("PSF map already made", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        infile_07_7 = mer_dir + "wide_broad_thresh.img"
        outfile_07_7 = mer_dir + "psfmap_wide.fits"
        try:
            sp.call(["bash", "shell/make_psf_map.sh", infile_07_7, outfile_07_7])
        except:
            print(
                colored(
                    "No image found in the 0.7-7 keV band!\n" + "--> Make image first",
                    "red",
                    None,
                    ["bold"],
                )
            )
            sys.exit()


def find_sources(res_dir, multiobs):
    """
    Finding point source regions using wavelets of different scales
    based on the image of the cluster in the 0.7-7 keV band

    Parameters
    __________
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Creates the region file containing the location of each point source

    """

    mer_dir = res_dir + "/results/"

    print(colored("Finding point sources...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "/wide_wdect_expmap_src.reg"):
        print(colored("Point sources already found", "white", None, ["bold"]))
        print("------------------------------------------------------------")
        return True
    else:
        map_file = mer_dir + "wide_broad_thresh.img"
        if multiobs:
            psf_file = mer_dir + "wide_broad_thresh.psfmap"
        else:
            psf_file = mer_dir + "psfmap_wide.fits"
        exp_file = mer_dir + "wide_broad_thresh.expmap"

        sp.call(
            ["bash", "shell/find_sources.sh", map_file, psf_file, exp_file, mer_dir]
        )

        source_file = mer_dir + "wide_wdect_expmap_src.reg"
        with open(source_file) as f:
            content = f.readlines()

        # Get the coordinates of each region
        coordinates_list = [content[i][8:-2].split(",") for i in range(len(content))]
        coordinates_arr = np.asarray(coordinates_list)
        shape_arr = coordinates_arr.shape

        if len(shape_arr) == 1:
            print(
                colored("No point source detected in the field", "grey", None, ["bold"])
            )
            print("------------------------------------------------------------")
            return False
        else:
            # Reshape the array so it's flat
            coordinates_arr_flat = coordinates_arr.reshape(coordinates_arr.size)
            # Convert each string into a float
            coordinates_arr_flat_float = [
                float(coordinates_arr_flat[i]) for i in range(coordinates_arr_flat.size)
            ]
            # Reshape array to original 2D shape
            coordinates_arr_float = np.asarray(coordinates_arr_flat_float).reshape(
                shape_arr[0], shape_arr[1]
            )
            # Find where the regions are flat
            wflat = np.where(coordinates_arr_float < 0.1)
            if len(wflat) > 0:
                # Replace null radii by finite ones
                coordinates_arr_float[wflat] = 2.0

                source_file2 = open(mer_dir + "wide_wdect_expmap_src2.reg", "w")
                for i in range(shape_arr[0]):
                    strbuff = ""
                    for j in range(shape_arr[1]):
                        strbuff += str(coordinates_arr_float[i, j]) + ","
                    source_file2.write("ellipse(" + strbuff[0:-1] + ")\n")

                source_file2.close()
                sp.call("rm " + source_file, shell=True)
                source_file2 = mer_dir + "wide_wdect_expmap_src2.reg"
                sp.call("mv " + source_file2 + " " + source_file, shell=True)
            return True


def check_sources(res_dir, is_sources):
    """
    Check the point source regions found in the image
    of the cluster in the 0.7-7 keV band

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    is_sources: are there sources in the image? True/False

    Returns
    _______
    Uses DS9 to show the cluster image along with the point source regions

    """

    mer_dir = res_dir + "/results/"

    print(colored("Checking point sources...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "/wide_wdect_expmap_src_check.fits"):
        print(colored("Point source regions already checked", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        if is_sources:
            map_file = mer_dir + "wide_broad_thresh.img"
            reg_file = mer_dir + "wide_wdect_expmap_src.reg"
            sp.call(["bash", "shell/ds9_cluster_region.sh", map_file, reg_file])
            time.sleep(6)
            sp.call(["bash", "shell/ds9_visual_improve.sh"])
            input("Add and/or modify source regions then press Enter to continue...")
            print("------------------------------------------------------------")
            reg_file_check = mer_dir + "wide_wdect_expmap_src_check.reg"
            sp.call(["bash", "shell/ds9_save_regions.sh", reg_file_check])
            time.sleep(3)
            reg_file_check_fits = mer_dir + "wide_wdect_expmap_src_check.fits"
            sp.call(
                [
                    "bash",
                    "shell/convert_region_file.sh",
                    map_file,
                    reg_file_check,
                    reg_file_check_fits,
                ]
            )
            sp.call("killall ds9", shell=True)
            print(colored("Point source regions saved", "grey", None, ["bold"]))
            print("------------------------------------------------------------")
        else:
            print(colored("No point source region to check", "grey", None, ["bold"]))
            print("------------------------------------------------------------")


def bkg_point_sources(res_dir, is_sources):
    """
    Finds the background regions associated with each
    point source detected in the 0.7-7 keV map

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    is_sources: are there sources in the image? True/False

    Returns
    _______
    A new directory in the results folder of res_dir called sources,
    containing all point source and associated background regions

    """

    mer_dir = res_dir + "/results/"
    reg_file_check_fits = mer_dir + "wide_wdect_expmap_src_check.fits"

    print(colored("Defining background for point sources...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "/exclude.bg.fits"):
        print(
            colored(
                "Point source background regions already defined",
                "white",
                None,
                ["bold"],
            )
        )
        print("------------------------------------------------------------")
    else:
        if is_sources:
            sp.call(
                ["bash", "shell/bkg_point_sources.sh", mer_dir, reg_file_check_fits]
            )
        else:
            print(colored("No point source region to use", "grey", None, ["bold"]))
            print("------------------------------------------------------------")


def subtract_point_sources(res_dir, is_sources, multiobs, obsids):
    """
    Creates an event file with point source excised and fill the
    holes in the cluster image in the 0.7-7 keV band

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    is_sources: are there sources in the image? True/False
    multiobs: are there multiple obsids to consider? True/False
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    A new event file with point source excised and a new cluster
    image in the 0.7-7 keV band without point sources

    """

    mer_dir = res_dir + "/results/"

    print(colored("Removing point sources...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "wide_broad_thresh_nopts.img") & os.path.exists(
        mer_dir + "efile_repro_raw_clean_nopts.fits"
    ):
        print(colored("Point sources already removed", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        if is_sources:
            reg_file = mer_dir + "/wide_wdect_expmap_src_check.reg"
            if multiobs:
                tab_obsid = obsids.split(",")
                for obsid in tab_obsid:
                    efile_in = mer_dir + "/All_" + obsid + "_reproj_evt.fits"
                    efile_out = mer_dir + "/All_" + obsid + "_reproj_evt_nopts.fits"
                    sp.call(
                        [
                            "bash",
                            "shell/efile_clean_pts.sh",
                            efile_in,
                            reg_file,
                            efile_out,
                        ]
                    )

            efile_in = mer_dir + "/efile_repro_raw_clean.fits"
            efile_out = mer_dir + "/efile_repro_raw_clean_nopts.fits"
            sp.call(["bash", "shell/efile_clean_pts.sh", efile_in, reg_file, efile_out])

            mapfile_in = mer_dir + "wide_broad_thresh.img"
            mapfile_out = mer_dir + "wide_broad_thresh_nopts.img"
            pts_reg = mer_dir + "/exclude.src.reg"
            pts_bkg_reg = mer_dir + "/exclude.bg.reg"
            sp.call(
                [
                    "bash",
                    "shell/subtract_pts_image.sh",
                    mapfile_in,
                    mapfile_out,
                    pts_reg,
                    pts_bkg_reg,
                ]
            )
        else:
            print(colored("No point source to remove", "grey", None, ["bold"]))
            print("------------------------------------------------------------")


def write_reg_centroid(cent_reg_name, X, Y, z, R500):
    """
    Creates a DS9 region file to find the centroid in an
    annulus region excluding the cluster core

    Parameters
    __________
    cent_reg_name: the file containing the DS9 region
    X: the RA position of the region center in pix coordinates
    X: the Dec position of the region center in pix coordinates
    z: the cluster redshift
    R500: the cluster R500 radius in pixel number

    Returns
    _______
    Creates a .reg file containing the extraction region

    """

    cent_reg = open(cent_reg_name, "w")
    cent_reg.write("# Region file format: CIAO version 1.0\n")
    if z < 0.1:
        cent_reg.write(
            "annulus("
            + str(X)
            + ","
            + str(Y)
            + ","
            + str(0.05 * R500)
            + ","
            + str(0.5 * R500)
            + ")"
        )
    else:
        cent_reg.write(
            "annulus("
            + str(X)
            + ","
            + str(Y)
            + ","
            + str(0.1 * R500)
            + ","
            + str(1.0 * R500)
            + ")"
        )
    cent_reg.close()
    time.sleep(1)


def get_cent_from_file(cent_file):
    """
    Extract the centroid location found by dmstat in the
    created file

    Parameters
    __________
    cent_file: the file created by dmstat containing the centroid

    Returns
    _______
    The X and Y positions of the centroid in pix coordinates

    """
    with open(cent_file) as f:
        content = f.readlines()
    centroid_X = float(content[4][15:-3].split(" ")[0])
    centroid_Y = float(content[4][15:-3].split(" ")[1])

    return [centroid_X, centroid_Y]


def find_peak_cent(res_dir, z, R500, use_peak, fixed_coord):
    """
    Finds the X-ray peak and centroid in 0.7-7 keV band image

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    z: the cluster redshift
    R500: the cluster R500 radius in kpc
    use_peak: boolean --> Use the X-ray peak as a deprojection center? True/False

    Returns
    _______
    Creates the file peak_cent_pos.txt containing the locations of both the X-ray peak
    and the X-ray centroid in the 0.7-7 keV band image and returns the
    deprojection center used in the ICM analysis

    """

    mer_dir = res_dir + "/results/"

    print(colored("Finding X-ray peak and centroid...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "peak_cent_pos.txt"):
        print(
            colored(
                "Deprojection center already found, restoring...",
                "white",
                None,
                ["bold"],
            )
        )
        print("------------------------------------------------------------")
        depro_center = ascii.read(mer_dir + "peak_cent_pos.txt")
    else:
        map_file = mer_dir + "wide_broad_thresh_nopts.img"

        hdu = fits.open(map_file)
        cl_map = hdu[0].data
        cl_header = hdu[0].header

        sig_smooth = 60.0
        pix_reso = hdu[0].header["CDELT2"] * 3600.0
        tophat_kernel = Tophat2DKernel(int(sig_smooth / pix_reso))
        map_conv = convolve(cl_map, tophat_kernel)

        Xpeak_img = map_conv.argmax() % map_conv.shape[1]
        Ypeak_img = map_conv.argmax() / map_conv.shape[1]

        N = 70
        small_map = cl_map[
            int(Ypeak_img - N) : int(Ypeak_img + N + 1),
            int(Xpeak_img - N) : int(Xpeak_img + N + 1),
        ]
        sig_smooth = 5.0
        tophat_kernel = Tophat2DKernel(int(sig_smooth / pix_reso))
        map_conv = convolve(small_map, tophat_kernel)

        Xpeak_img_fine = map_conv.argmax() % map_conv.shape[1]
        Ypeak_img_fine = map_conv.argmax() / map_conv.shape[1]

        Xpeak = (Xpeak_img + (Xpeak_img_fine - N) - cl_header["CRPIX1P"]) * cl_header[
            "CDELT1P"
        ] + cl_header["CRVAL1P"]
        Ypeak = (Ypeak_img + (Ypeak_img_fine - N) - cl_header["CRPIX2P"]) * cl_header[
            "CDELT2P"
        ] + cl_header["CRVAL2P"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = wcs.WCS(cl_header)
            get_coord = w.wcs_pix2world(
                float(Xpeak_img + (Xpeak_img_fine - N)),
                float(Ypeak_img + (Ypeak_img_fine - N)),
                1,
            )
            Xpeak_coord = float(get_coord[0])
            Ypeak_coord = float(get_coord[1])

        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        R500_pix = (
            ((R500 / d_a) * u.rad).to("arcsec")
            / (cl_header["CDELT2"] * 3600.0 / cl_header["CDELT2P"])
        ).value

        cent_reg_name = mer_dir + "find_centroid.reg"
        write_reg_centroid(cent_reg_name, Xpeak, Ypeak, z, R500_pix)
        cent_file = mer_dir + "centroid_location.txt"
        sp.call(["bash", "shell/find_centroid.sh", map_file, cent_reg_name, cent_file])
        Xcent, Ycent = get_cent_from_file(cent_file)
        write_reg_centroid(cent_reg_name, Xcent, Ycent, z, R500_pix)
        sp.call(["bash", "shell/find_centroid.sh", map_file, cent_reg_name, cent_file])
        Xcent, Ycent = get_cent_from_file(cent_file)

        Xcent_img = (float(Xcent) - cl_header["CRVAL1P"]) / cl_header[
            "CDELT1P"
        ] + cl_header["CRPIX1P"]
        Ycent_img = (float(Ycent) - cl_header["CRVAL2P"]) / cl_header[
            "CDELT2P"
        ] + cl_header["CRPIX2P"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            get_coord = w.wcs_pix2world(float(Xcent_img), float(Ycent_img), 1)
            Xcent_coord = float(get_coord[0])
            Ycent_coord = float(get_coord[1])

        if fixed_coord is not None:
            Xpeak_coord = fixed_coord[0]
            Ypeak_coord = fixed_coord[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                get_pix = w.wcs_world2pix(Xpeak_coord, Ypeak_coord, 1)
                Xpeak_img = float(get_pix[0])
                Ypeak_img = float(get_pix[1])
                Xpeak = (Xpeak_img - cl_header["CRPIX1P"]) * cl_header[
                    "CDELT1P"
                ] + cl_header["CRVAL1P"]
                Ypeak = (Ypeak_img - cl_header["CRPIX2P"]) * cl_header[
                    "CDELT2P"
                ] + cl_header["CRVAL2P"]

        depro_center = {
            "X_Y_peak": [Xpeak, Ypeak],
            "X_Y_peak_coord": [Xpeak_coord, Ypeak_coord],
            "X_Y_cent": [Xcent, Ycent],
            "X_Y_cent_coord": [Xcent_coord, Ycent_coord],
        }

        depro_center = Table(depro_center)
        ascii.write(depro_center, mer_dir + "peak_cent_pos.txt", overwrite=True)

    if use_peak:
        return [depro_center["X_Y_peak"][0], depro_center["X_Y_peak"][1]]
    else:
        return [depro_center["X_Y_cent"][0], depro_center["X_Y_cent"][1]]


def bkg_region(res_dir, z, R500, Xdepro, Ydepro, multiobs, obsids):
    """
    Computes the background region far from the cluster

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    z: the cluster redshift
    R500: the cluster R500 radius in kpc
    Xdepro: the RA position of the deprojection center
    Ydepro: the Dec position of the deprojection center
    multiobs: are there multiple obsids to consider? True/False
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates a DS9 region file called bkg_region.reg from which the
    background can be estimated. Returns the area of the region
    in pixel**2

    """

    mer_dir = res_dir + "/results/"

    print(colored("Defining background region...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    tab_obsid = obsids.split(",")
    bkg_area_tab = []

    if os.path.exists(mer_dir + "bkg_region_" + tab_obsid[0] + ".reg"):
        print(colored("Background region already defined", "white", None, ["bold"]))
        print("------------------------------------------------------------")
        for obsid in tab_obsid:
            file_area = mer_dir + "bkg_area_" + obsid + ".txt"
            with open(file_area) as f:
                content = f.readlines()
            bkg_area = float(content[0])
            bkg_area_tab.append(bkg_area)
    else:
        efile = mer_dir + "efile_repro_raw_clean_nopts.fits"
        for obsid in tab_obsid:
            if multiobs:
                map_file = mer_dir + "wide_" + obsid + "_broad_thresh.img"
            else:
                map_file = mer_dir + "wide_broad_thresh_nopts.img"

            hdu = fits.open(map_file)
            cl_header = hdu[0].header

            roll_angle = cl_header["ROLL_PNT"]

            if roll_angle < 180.0:
                start_angle = 360.0 - roll_angle
            else:
                start_angle = 180.0 + (360.0 - roll_angle)

            d_a = cosmo.angular_diameter_distance(z).to("kpc").value
            R500_pix = (
                ((R500 / d_a) * u.rad).to("arcsec")
                / (cl_header["CDELT2"] * 3600.0 / cl_header["CDELT2P"])
            ).value
            chip_width = (8.3 * u.arcmin).to("arcsec").value
            chip_width_pix = chip_width / (
                cl_header["CDELT2"] * 3600.0 / cl_header["CDELT2P"]
            )

            Xmid = (
                cl_header["CRVAL1P"] + (cl_header["NAXIS1"] * cl_header["CDELT2P"]) / 2
            )
            Ymid = (
                cl_header["CRVAL2P"] + (cl_header["NAXIS2"] * cl_header["CDELT2P"]) / 2
            )
            delta_depro_mid = np.sqrt((Xmid - Xdepro) ** 2 + (Ymid - Ydepro) ** 2)
            delta2chip = delta_depro_mid * np.sin(
                45.0 * np.pi / 180.0
            )  # cluster is on the diagonal of a chip
            outer_rad = delta2chip + chip_width_pix
            inner_rad = 1.5 * R500_pix
            if outer_rad < inner_rad:
                inner_rad = (
                    outer_rad
                    - (
                        (1.0 * u.arcmin).to("arcsec")
                        / (cl_header["CDELT2"] * 3600.0 / cl_header["CDELT2P"])
                    ).value
                )

            reg_file_cl = mer_dir + "bkg_region_" + obsid + ".reg"
            file_bkg = open(reg_file_cl, "w")
            file_bkg.write("# Region file format: CIAO version 1.0\n")
            file_bkg.write(
                "pie("
                + str(Xdepro)
                + ","
                + str(Ydepro)
                + ","
                + str(inner_rad)
                + ","
                + str(outer_rad)
                + ","
                + str(start_angle)
                + ","
                + str(start_angle + 90)
                + ")"
            )
            file_bkg.close()

            efile_in = efile + "[bin sky=@" + reg_file_cl + "]"
            file_out = mer_dir + "bkg_stat_" + obsid + ".fits"
            sp.call(["bash", "shell/extract_content.sh", efile_in, file_out])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hdu = fits.open(file_out)
                bkg_area = hdu[1].data["AREA"][0]
                bkg_area_tab.append(bkg_area)
                file_area = open(mer_dir + "bkg_area_" + obsid + ".txt", "w")
                file_area.write(str(bkg_area))
                file_area.close()

    return bkg_area_tab


def find_SB_annuli(res_dir, Xdepro, Ydepro, bkg_area, z, R500, fast_annuli, obsids):
    """
    Defines the annuli for the X-ray surface brightness profile based
    on the S/N in each annulus or with the polynomial definition
    given in McDonald et al. 2017 (https://arxiv.org/pdf/1702.05094.pdf)

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    Xdepro: the RA position of the deprojection center
    Ydepro: the Dec position of the deprojection center
    bkg_area: the area of the background region in pixel**2
    z: the cluster redshift
    R500: the cluster R500 radius in kpc
    fast_annuli: boolean --> Fast computation of the SB annuli? True/False
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates a DS9 region file called SB_annuli.reg to be used to
    extract the X-ray surface brightness profile

    """

    mer_dir = res_dir + "/results/"

    print(
        colored(
            "Defining annuli for surface brightness profile...", "blue", None, ["bold"]
        )
    )
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "SB_annuli.reg"):
        print(
            colored(
                "Surface brightness annuli already defined", "white", None, ["bold"]
            )
        )
        print("------------------------------------------------------------")
    else:
        tab_obsid = obsids.split(",")
        map_file = mer_dir + "wide_broad_thresh_nopts.img"
        hdu = fits.open(map_file)
        cl_header = hdu[0].header
        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        R500_pix = (
            ((R500 / d_a) * u.rad).to("arcsec")
            / (cl_header["CDELT2"] * 3600.0 / cl_header["CDELT2P"])
        ).value

        if fast_annuli:
            tab_rad = []
            for i in range(1, 17):
                tab_rad.append(
                    (13.779 - 8.8148 * i + 7.2829 * i ** 2 - 0.15633 * i ** 3)
                    * 1e-3
                    * R500_pix
                )
            tab_rad.insert(0, 0.0)
            reg_file_name = mer_dir + "SB_annuli.reg"
            reg_file = open(reg_file_name, "w")
            reg_file.write("# Region file format: CIAO version 1.0\n")
            for i in range(16):
                reg_file.write(
                    "annulus("
                    + str(Xdepro)
                    + ","
                    + str(Ydepro)
                    + ","
                    + str(tab_rad[i])
                    + ","
                    + str(tab_rad[i + 1])
                    + ")\n"
                )
            reg_file.close()
        else:
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
            reg_file = mer_dir + "bkg_region_" + tab_obsid[0] + ".reg"
            bkg_count_file = mer_dir + "bkg_counts.txt"
            sp.call(
                ["bash", "shell/counts_in_reg.sh", out_file, reg_file, bkg_count_file]
            )
            with open(bkg_count_file) as f:
                content = f.readlines()
            bkg_counts = float(content[5][9:-1])
            bkg_count_rate = bkg_counts / cl_header["exposure"] / bkg_area[0]

            # Loop to find annuli given S/N per annulus
            reg_file_name_i = mer_dir + "SB_annulus_i.reg"
            counts_file_name_i = mer_dir + "SB_annulus_counts_i.txt"
            inner_rad = 0.0
            outer_rad = 0.0
            Delta_r = 0.0
            inner_rad_tab = []
            outer_rad_tab = []
            while (outer_rad < (1.5 * R500_pix)) & (Delta_r < 2.0):
                rad_add = 0.0
                S2N = 0.0
                Delta_r = 0.0
                while (S2N < 5) & (Delta_r < 2.0):
                    rad_add += 4.0
                    outer_rad = inner_rad + rad_add
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
                    Delta_r = rad_add * (
                        cl_header["CDELT2"] * 60.0 / cl_header["CDELT2P"]
                    )

                inner_rad_tab.append(inner_rad)
                outer_rad_tab.append(outer_rad)
                inner_rad += rad_add

            reg_file_name = mer_dir + "SB_annuli.reg"
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


def vignetting_prof(res_dir, obsids):
    """
    Computes the weights to apply to the X-ray surface brightness
    profile computed in each bin in SB_annuli.reg and to the background value

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates a .npz file containing two numpy arrays. The first one contains
    the weights to be applied to the surface brightness profile. The second
    one contains the single weight to apply to the background value

    """

    mer_dir = res_dir + "/results/"

    print(colored("Computing vignetting profile...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "vignetting_prof.npz"):
        print(colored("Vignetting profile already computed", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        expfile = mer_dir + "SB_0.7-2_thresh.expmap"
        hdu = fits.open(expfile)
        expmap = hdu[0].data
        expmax = np.max(expmap)

        annuli_file = mer_dir + "/SB_annuli.reg"
        with open(annuli_file) as f:
            content = f.readlines()

        regfile_i = mer_dir + "SB_annuli_i.reg"
        exp_in_i = mer_dir + "exp_in_i.txt"
        vignetting_prof = np.zeros(len(content) - 1)

        for i in range(1, len(content)):
            file_annuli_i = open(regfile_i, "w")
            file_annuli_i.write("# Region file format: CIAO version 1.0\n")
            file_annuli_i.write(content[i][:-1])
            file_annuli_i.close()
            sp.call(["bash", "shell/counts_in_reg.sh", expfile, regfile_i, exp_in_i])

            with open(exp_in_i) as f:
                vign_cl = f.readlines()

            vignetting_prof[i - 1] = float(vign_cl[3][9:-1]) / expmax

        tab_obsid = obsids.split(",")
        regfile_bkg = mer_dir + "bkg_region_" + tab_obsid[0] + ".reg"
        exp_in_bkg = mer_dir + "exp_in_bkg.txt"
        sp.call(["bash", "shell/counts_in_reg.sh", expfile, regfile_bkg, exp_in_bkg])
        with open(exp_in_bkg) as f:
            content = f.readlines()

        vignetting_bkg = float(content[3][9:-1]) / expmax

        saved_vign_prof = mer_dir + "vignetting_prof.npz"
        np.savez(saved_vign_prof, cl=vignetting_prof, bkg=vignetting_bkg)


def X_ray_SB_profile(res_dir, obsids, z):
    """
    Computes the vignetted-corrected X-ray surface brightness profile
    and save it in a .npz file

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates a .npz file containing the X-ray surface brightness profile
    along with its associated uncertainties

    """

    mer_dir = res_dir + "/results/"

    print(colored("Computing surface brightness profile...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(mer_dir + "cl_bkg_sb_profile.npz"):
        print(
            colored(
                "Surface brightness profile already computed", "white", None, ["bold"]
            )
        )
        print("------------------------------------------------------------")
    else:
        tab_obsid = obsids.split(",")
        efile = mer_dir + "efile_repro_raw_clean_nopts.fits"
        reg_file_cl = mer_dir + "SB_annuli.reg"
        reg_file_bkg = mer_dir + "bkg_region_" + tab_obsid[0] + ".reg"
        cl_file = efile + "[bin sky=@" + reg_file_cl + "][energy=700:2000]"
        bkg_file = efile + "[bin sky=@" + reg_file_bkg + "][energy=700:2000]"
        out_file = mer_dir + "XSB_profile.fits"
        out_file_rmid = mer_dir + "XSB_profile_rmid.fits"

        sp.call(
            ["bash", "shell/XSB_profile.sh", cl_file, bkg_file, out_file, out_file_rmid]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdu = fits.open(out_file_rmid)
            prof = hdu[1].data
            header = hdu[1].header

        vignetting_file = mer_dir + "vignetting_prof.npz"
        vig_data = np.load(vignetting_file)
        vignetting_prof = vig_data["cl"]
        vignetting_bkg = float(vig_data["bkg"])

        Rmid = prof["rmid"] * header["TCDLT6"] / 60.0
        XSB_prof_N = prof["counts"] / vignetting_prof
        XSB_prof_err_N = prof["err_counts"] / vignetting_prof
        XSB_prof_bkg_N = (prof["bg_counts"] / vignetting_bkg / prof["bg_area"]) * prof[
            "area"
        ]
        XSB_prof_bkg_err_N = (prof["bg_err"] / vignetting_bkg / prof["bg_area"]) * prof[
            "area"
        ]
        XSB_float_to_N = 3600.0 * header["TCDLT21"] / prof["area"] / prof["exposure"]

        saved_xsb_prof = mer_dir + "cl_bkg_sb_profile.npz"
        np.savez(
            saved_xsb_prof,
            r=Rmid,
            xsb=XSB_prof_N,
            xsb_err=XSB_prof_err_N,
            xsb_bkg=XSB_prof_bkg_N,
            xsb_bkg_err=XSB_prof_bkg_err_N,
            f2n=XSB_float_to_N,
        )

        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        Rmid_kpc = d_a * (Rmid * u.arcmin).to("radian").value

        fits.writeto(mer_dir + "Sx_profile_MCMC.fits", Rmid, overwrite=True)
        fits.append(mer_dir + "Sx_profile_MCMC.fits", Rmid_kpc, overwrite=True)
        fits.append(mer_dir + "Sx_profile_MCMC.fits", XSB_prof_N, overwrite=True)
        fits.append(mer_dir + "Sx_profile_MCMC.fits", XSB_prof_err_N, overwrite=True)
        fits.append(mer_dir + "Sx_profile_MCMC.fits", XSB_prof_bkg_N, overwrite=True)
        fits.append(mer_dir + "Sx_profile_MCMC.fits", XSB_prof_bkg_err_N, overwrite=True)
        fits.append(mer_dir + "Sx_profile_MCMC.fits", XSB_float_to_N, overwrite=True)
