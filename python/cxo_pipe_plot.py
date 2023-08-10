import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import warnings
import subprocess as sp
import matplotlib as mpl
import os
from termcolor import colored
from getdist import plots, MCSamples
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
import matplotlib.colors as col
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib import ticker
import astropy.constants as const
import scipy.ndimage

cosmo = FlatLambdaCDM(70.0, 0.3, Tcmb0=2.7255)

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
mpl.rcParams["text.latex.preamble"] = [r"\boldmath"]
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"


def crop_fits_to_FoV(in_fits, out_fits, center, FoV):
    """
    Crops the HDU of an input .fits file to a specified
    FoV around a central point

    Parameters
    __________
    in_fits: path to a .fits file
    center: (RA, dec) tuple of the object in degrees,
    FoV: the size you want your map, in arcminutes

    Returns
    _______
    A .fits file with the HDU cropped and header adapted
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdulist = fits.open(in_fits)
        center_coords = SkyCoord(ra=center[0] * u.deg, dec=center[1] * u.deg)

        hdu = hdulist[0]
        data, header = hdu.data, hdu.header
        wcs = WCS(header, fix=True)

        size = u.Quantity((FoV, FoV), u.arcmin)

        cutout = Cutout2D(data, center_coords, size, wcs=wcs)

        new_hdu = hdu
        new_hdu.data = cutout.data
        new_hdu.header.update(cutout.wcs.to_header())

        fits.HDUList(new_hdu).writeto(out_fits, overwrite=True)


def hex_to_rgb(tab_value):

    tab_rgb = []
    for i in range(len(tab_value)):
        value = tab_value[i].lstrip("#")
        lv = len(value)
        tab_rgb.append(
            tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
        )
    return tab_rgb


def make_cmap(colors, position=None, bit=False):
    """
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    """
    import matplotlib as mpl
    import numpy as np

    bit_rgb = np.linspace(0, 1, 256)
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (
                bit_rgb[colors[i][0]],
                bit_rgb[colors[i][1]],
                bit_rgb[colors[i][2]],
            )
    cdict = {"red": [], "green": [], "blue": []}
    for pos, color in zip(position, colors):
        cdict["red"].append((pos, color[0], color[0]))
        cdict["green"].append((pos, color[1], color[1]))
        cdict["blue"].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap("my_colormap", cdict, 256)
    return cmap


def map_display(
    mapfile,
    save,
    beam=0,
    beam_offset=0,
    peak=None,
    centroid=None,
    colorbar=1,
    c_title="",
    ruler=0,
    ruler_label="",
    mincol=None,
):
    """
    Plot a map from a .fits file

    Parameters
    __________
    mapfile: the input .fits file with the map in entry 0
    save: the output .pdf file name
    beam: beam size in arcsec
    beam_offset: offset of the beam location from bottom-left corner
    peak: coordinates of the peak location
    centroid: coordinates of the centroid location
    colobar: do you want to display a color bar?
    c_title: colorbar label
    ruler: size of the ruler in arcsec
    ruler_label: ruler label (usually the corresponding physical size)

    Returns
    _______
    A .fits file with the HDU cropped and header adapted
    """

    plt.close("all")

    colors = hex_to_rgb(
        [
            "#000614",
            "#140144",
            "#280283",
            "#912f7f",
            "#f05d79",
            "#f3a35d",
            "#f7ef43",
            "#CCCBB2",
            "#E3E3D5",
            "#FBFBF3",
            "#FFFFFF",
        ]
    )
    position = np.linspace(0, 1, 11)
    cmap = make_cmap(colors, position=position, bit=True)

    hdulist = fits.open(mapfile)
    wcs = (WCS(hdulist[0].header)).celestial
    image_plot = hdulist[0].data

    if (np.asarray(image_plot.shape)).size != 2:
        nx = hdulist[0].header["NAXIS2"]
        ny = hdulist[0].header["NAXIS1"]
        image_plot = np.reshape(image_plot, (nx, ny))

    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(8.7)
    ax = fig.add_subplot(1, 1, 1, projection=wcs)

    for tick in ax.get_xticklabels():
        tick.set_fontname("Helvetica")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Helvetica")

    if mincol is not None:
        vmin = mincol
    else:
        vmin = image_plot.min()
    vmax = image_plot.max()

    plt.imshow(
        image_plot, cmap=cmap, origin="lower", norm=col.LogNorm(vmin=vmin, vmax=vmax)
    )

    plt.rc("text", usetex=True)

    if (
        (
            (hdulist[0].header["CTYPE1"] == "RA---TAN")
            & (hdulist[0].header["CTYPE2"] == "DEC--TAN")
        )
        | (
            (hdulist[0].header["CTYPE1"] == "RA---SIN")
            & (hdulist[0].header["CTYPE2"] == "DEC--SIN")
        )
        | (
            (hdulist[0].header["CTYPE1"] == "RA---SFL")
            & (hdulist[0].header["CTYPE2"] == "DEC--SFL")
        )
    ):
        plt.xlabel(r"$\mathrm{Right~Ascension~(J2000)~[hr]}$", fontsize=15)
        plt.ylabel(r"$\mathrm{Declination~(J2000)~[degree]}$", fontsize=15)
        lon = ax.coords["ra"]
        lat = ax.coords["dec"]
        lon.set_ticks_position("b")
        lat.set_ticks_position("l")
        lon.set_major_formatter("hh:mm:ss")
        lat.set_major_formatter("dd:mm:ss")

    ax.coords[0].set_ticklabel(color="black", size=14, weight="bold")
    ax.coords[1].set_ticklabel(color="black", size=14, weight="bold")

    if colorbar == 1:
        cbar = plt.colorbar(pad=0.02, aspect=26)
        cbar.set_label(r"%s" % (c_title), labelpad=10, size=12)

    if beam != 0:
        hdr = hdulist[0].header
        if "CDELT2" in hdr:
            pix_size = hdulist[0].header["CDELT2"] * 3600.0
        else:
            pix_size = hdulist[0].header["CD1_1"] * 3600.0
        beam_plot = Ellipse(
            (
                0.65 * beam / pix_size + beam_offset / pix_size,
                0.65 * beam / pix_size + beam_offset / pix_size,
            ),
            beam / pix_size,
            beam / pix_size,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(beam_plot)

    if peak is not None:
        ax.scatter(
            peak[0],
            peak[1],
            transform=ax.get_transform("fk5"),
            s=30,
            edgecolor="white",
            facecolor="red",
            zorder=10000,
        )

    if centroid is not None:
        ax.scatter(
            centroid[0],
            centroid[1],
            transform=ax.get_transform("fk5"),
            s=30,
            edgecolor="white",
            facecolor="blue",
            zorder=10000,
        )

    if ruler != 0:
        xmin, xmax = [0, image_plot.shape[0]]
        ymin, ymax = [0, image_plot.shape[1]]
        pix_size = hdulist[0].header["CDELT2"] * 3600.0
        Nx = hdulist[0].header["NAXIS1"]
        r = Rectangle(
            (
                xmax - 0.05 * (xmax - xmin) - (ruler / pix_size),
                ymin + 0.05 * (ymax - ymin),
            ),
            ruler / pix_size,
            0.0075 * (ymax - ymin),
            edgecolor="white",
            facecolor="white",
        )
        ax.add_patch(r)

    if ruler_label != "":
        ax.text(
            xmax - 0.05 * (xmax - xmin) - (ruler / pix_size) / 2.0,
            ymin + 0.022 * (ymax - ymin),
            r"\textbf{%s}" % (ruler_label),
            ha="center",
            va="center",
            size=8,
            weight="bold",
            color="white",
        )

    plt.savefig(save)

    plt.close("all")


def plot_spectra(res_dir, obsids):
    """
    Plot the cluster and background spectra and their
    associated best-fit models given by fit_spec in cxo_pipe_spec

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    obsids: the list of obsids given as a comma-separated string of numbers

    Returns
    _______
    Creates .pdf files in the *figures* folder of the *results*
    directory in res_dir showing the spectrum extracted in each
    annulus and the corresponding best-fit model

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    fig_dir = mer_dir + "figures/"

    if not os.path.exists(fig_dir):
        sp.call("mkdir " + fig_dir, shell=True)

    print(colored("Plotting spectra...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    fig_files = glob.glob(fig_dir + "spectrum_*.pdf")

    if len(fig_files) > 0:
        print(colored("Spectra already plotted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:

        tab_obsid = obsids.split(",")
        spec_files = glob.glob(cl_dir + "spec_fit_" + tab_obsid[0] + "_*.npz")

        for obsid in tab_obsid:
            for i in range(1, len(spec_files) + 1):
                spec_ann_file = cl_dir + "spec_fit_" + obsid + "_" + str(i) + ".npz"
                spec_ann = np.load(spec_ann_file)
                int_bkg_fit = np.interp(
                    spec_ann["fitx"], spec_ann["bkgfitx"], spec_ann["bkgfity"]
                )
                file_save = fig_dir + "spectrum_" + obsid + "_" + str(i) + ".pdf"
                with PdfPages(file_save) as pdf:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fig, ax = plt.subplots(nrows=1, sharex=True)
                        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 3])
                        ax0 = plt.subplot(gs[0])
                        ax1 = plt.subplot(gs[1])
                        gs.update(hspace=0.0)
                        ax0.set_yscale("log")
                        plt.rc("text", usetex=True)
                        plt.rcParams["text.latex.preamble"] = [r"\boldmath"]
                        ax0.set_xlim(0.7, 9)
                        ax0.set_ylabel(
                            r"$\mathrm{Counts~s^{-1}~keV^{-1}}$", fontsize=12
                        )
                        ax0.set_ylim(
                            np.max(
                                [
                                    1e-5,
                                    0.9
                                    * np.min(
                                        float(spec_ann["bkgsc"])
                                        * int_bkg_fit[spec_ann["fitx"] < 7.0]
                                    ),
                                ]
                            ),
                            1.1 * np.max(spec_ann["datay"] + spec_ann["datayerr"]),
                        )
                        ax0.grid(alpha=0.5, which="both")
                        ax0.set_xticklabels([])
                        ax0.plot(
                            spec_ann["fitx"],
                            spec_ann["fity"],
                            color="#BA094A",
                            lw=1.7,
                            label=r"$\mathrm{Total}$",
                            zorder=10,
                        )
                        ax0.plot(
                            spec_ann["fitx"],
                            spec_ann["fity"] - float(spec_ann["bkgsc"]) * int_bkg_fit,
                            color="#ED591A",
                            lw=1.4,
                            ls=(0, (3, 5)),
                            label=r"$\mathrm{Source}$",
                            dash_capstyle="round",
                            zorder=9,
                        )
                        ax0.plot(
                            spec_ann["bkgfitx"],
                            float(spec_ann["bkgsc"]) * spec_ann["bkgfity"],
                            color="#8C6873",
                            lw=1.4,
                            ls=(0, (3, 5, 1, 5, 1, 5)),
                            label=r"$\mathrm{Background}$",
                            dash_capstyle="round",
                            zorder=8,
                        )
                        ax0.errorbar(
                            spec_ann["datax"],
                            spec_ann["datay"],
                            yerr=spec_ann["datayerr"],
                            xerr=spec_ann["dataxerr"],
                            fmt="o",
                            label=r"$\mathrm{Data}$",
                            mfc="#31B0CC",
                            mec="#001199",
                            mew=1.5,
                            ecolor="#001199",
                            elinewidth=1,
                            capthick=1,
                        )

                        ax1.plot(
                            spec_ann["bkgfitx"],
                            spec_ann["bkgfity"],
                            color="#BA094A",
                            lw=1.2,
                            label=r"$\mathrm{Background~model}$",
                            zorder=10,
                        )
                        ax1.errorbar(
                            spec_ann["bkgdatx"],
                            spec_ann["bkgdaty"],
                            yerr=spec_ann["bkgdatyerr"],
                            xerr=spec_ann["bkgdatxerr"],
                            fmt="o",
                            label=r"$\mathrm{Background~data}$",
                            mfc="#31B0CC",
                            mec="#001199",
                            mew=1.5,
                            ecolor="#001199",
                            elinewidth=1,
                            capthick=1,
                        )
                        ax1.set_ylim(
                            0.9
                            * np.min(
                                (spec_ann["bkgdaty"] - spec_ann["bkgdatyerr"])[
                                    spec_ann["bkgdatx"] < 7.0
                                ]
                            ),
                            1.1 * np.max(spec_ann["bkgdaty"] + spec_ann["bkgdatyerr"]),
                        )
                        ax1.set_xlim(0.7, 9)
                        ax1.set_xlabel(r"$\mathrm{Energy~[keV]}$", fontsize=12)
                        ax1.set_ylabel(
                            r"$\mathrm{Counts~s^{-1}~keV^{-1}}$", fontsize=12
                        )
                        ax1.grid(alpha=0.5, which="both")
                        if (
                            np.max(spec_ann["fity"])
                            / np.max(float(spec_ann["bkgsc"]) * spec_ann["bkgfity"])
                            > 10
                        ):
                            leg = ax0.legend(fontsize=10, loc=4, framealpha=1)
                        else:
                            leg = ax0.legend(fontsize=10, loc=1, framealpha=1)
                        leg.set_zorder(100)
                        ax1.legend(fontsize=10, loc=1)
                        pdf.savefig()
                        plt.close()


def plot_T_prof(res_dir, R500):
    """
    Plot the spectroscopic temperature profile and
    its associated best-fit using the Vikhlinin+2006 model

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    R500: the cluster R500 radius in kpc

    Returns
    _______
    Creates a .pdf file in the *figures* folder of the *results*
    directory in res_dir showing the ICM temperature profile

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    fig_dir = mer_dir + "figures/"

    print(colored("Plotting temperature profile...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_save = fig_dir + "T_profile.pdf"

    if os.path.exists(file_save):
        print(colored("Temperature profile already plotted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        T_prof_file = cl_dir + "T_prof_fit.npz"
        T_data = np.load(T_prof_file)

        mean_ICM_T = np.mean(T_data["datay"])
        if T_data["datay"].size > 1:
            std_ICM_T = np.std(T_data["datay"]) / np.sqrt(T_data["datay"].size)
        else:
            std_ICM_T = np.mean(T_data["datayerr"])

        with PdfPages(file_save) as pdf:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig, ax = plt.subplots(nrows=1, sharex=True)
                ax.set_xscale("log")
                plt.xlabel(r"$\mathrm{Radius~[kpc]}$", fontsize=12)
                plt.ylabel(r"$\mathrm{Temperature~T_X~[keV]}$", fontsize=12)
                plt.grid(True, alpha=0.5, which="both")
                plt.xlim([10.0, 2.0 * R500])
                plt.ylim([0.0, 20.0])
                T_high = T_data["fity"] + T_data["fityerr"]
                T_low = T_data["fity"] - T_data["fityerr"]
                ax.fill_between(
                    T_data["fitx"],
                    T_low,
                    T_high,
                    where=T_high >= T_low,
                    facecolor="#89C8E8",
                    edgecolor="#89C8E8",
                    interpolate=True,
                    alpha=0.8,
                )
                plt.plot(
                    T_data["fitx"],
                    T_data["fity"],
                    color="#051F3B",
                    lw=1.5,
                    label=r"$\mathrm{V06~model}$",
                )
                plt.errorbar(
                    T_data["datax"],
                    T_data["datay"],
                    yerr=T_data["datayerr"],
                    xerr=T_data["dataxerr"],
                    fmt="o",
                    label=r"$\mathrm{T_X = }"
                    + "{:4.2f}".format(mean_ICM_T)
                    + " \pm "
                    + "{:4.2f}".format(std_ICM_T)
                    + "$",
                    mfc="#31B0CC",
                    mec="#001199",
                    mew=1.5,
                    ecolor="#001199",
                    elinewidth=1,
                    capthick=1,
                )
                plt.legend(fontsize=12, loc=2)
                pdf.savefig()
                plt.close()


def profile_plot(
    file_name,
    x,
    y,
    yerru,
    yerrd,
    title_dict,
    frame_dict,
    legend_dict=None,
    data_dict=None,
    A_dict=None,
):
    """
    Creates a .pdf file showing the profile given
    in argument with its associated uncertainties
    at 1 and 2-sigma. Data can also be over-plotted
    on the profile.

    Parameters
    __________
    file_name: the name of the .pdf file
    x: the array in the abscissa direction
    y: the array in the ordinate direction
    yerru, yerrd: the asymmetric error bars associated with y
    title_dict: a dictionnary containing the titles in both directions
    frame_dict: a dictionnary containing the frame parameters
    legend_dict: the labels associated with each curve
    data_dict: a dictionnary containing the data to be over-plotted
    A_dict: a dictionnary containing the ACCEPT results to be over-plotted

    Returns
    _______
    Creates a .pdf file in the *figures* folder of the *results*
    directory in res_dir showing the considered profile

    """

    with PdfPages(file_name) as pdf:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(7, 5))
            if frame_dict["logx"]:
                ax.set_xscale("log")
            if frame_dict["logy"]:
                ax.set_yscale("log")
            plt.xlim(frame_dict["xr"])
            plt.ylim(frame_dict["yr"])
            plt.xlabel(title_dict["xlab"], fontsize=17)
            plt.ylabel(title_dict["ylab"], fontsize=17)
            ax.xaxis.set_tick_params(width=1.5)
            ax.yaxis.set_tick_params(width=1.5)
            ax.xaxis.set_tick_params(labelsize=15)
            ax.yaxis.set_tick_params(labelsize=15)
            plt.grid(True, alpha=0.3, which="both")
            y_high = (
                y[np.argwhere(~np.isnan(yerru))[:, 0]]
                + yerru[np.argwhere(~np.isnan(yerru))[:, 0]]
            )
            y_low = (
                y[np.argwhere(~np.isnan(yerru))[:, 0]]
                - yerrd[np.argwhere(~np.isnan(yerru))[:, 0]]
            )
            y_2high = (
                y[np.argwhere(~np.isnan(yerru))[:, 0]]
                + 2.0 * yerru[np.argwhere(~np.isnan(yerru))[:, 0]]
            )
            y_2low = (
                y[np.argwhere(~np.isnan(yerru))[:, 0]]
                - 2.0 * yerrd[np.argwhere(~np.isnan(yerru))[:, 0]]
            )
            ax.fill_between(
                x[np.argwhere(~np.isnan(yerru))[:, 0]],
                y_2low,
                y_2high,
                where=y_2high >= y_2low,
                facecolor="#A8C2D8",
                edgecolor="#A8C2D8",
                interpolate=True,
                zorder=2,
                alpha=0.8,
            )
            ax.fill_between(
                x[np.argwhere(~np.isnan(yerru))[:, 0]],
                y_low,
                y_high,
                where=y_high >= y_low,
                facecolor="#6188AA",
                edgecolor="#6188AA",
                interpolate=True,
                zorder=2,
                alpha=0.8,
            )
            leg_fill = ax.fill(np.NaN, np.NaN, "#6188AA", alpha=0.5, linewidth=0)

            if data_dict is not None:
                if data_dict["xerr"] is not None:
                    leg_pt1 = plt.errorbar(
                        data_dict["x"],
                        data_dict["y"],
                        xerr=data_dict["xerr"],
                        yerr=data_dict["yerr"],
                        fmt="o",
                        mfc="#CA89A8",
                        mec="#7B1E4A",
                        mew=1.5,
                        ecolor="#7B1E4A",
                        elinewidth=1,
                        capthick=1,
                    )
                else:
                    leg_pt1 = plt.errorbar(
                        data_dict["x"],
                        data_dict["y"],
                        yerr=data_dict["yerr"],
                        fmt="o",
                        mfc="#CA89A8",
                        mec="#7B1E4A",
                        mew=1.5,
                        ecolor="#7B1E4A",
                        elinewidth=1,
                        capthick=1,
                    )

            if A_dict is not None:
                leg_pt2 = plt.errorbar(
                    A_dict["x"],
                    A_dict["y"],
                    yerr=A_dict["yerr"],
                    fmt="^",
                    mfc="#E9B490",
                    mec="#D26315",
                    mew=1.5,
                    ecolor="#D26315",
                    elinewidth=1,
                    capthick=1,
                )

            leg_line = plt.plot(x, y, color="#193854", lw=1.5, zorder=3)

            if (data_dict is not None) & (A_dict is not None):
                ax.legend(
                    [(leg_fill[0], leg_line[0]), leg_pt1, leg_pt2],
                    [legend_dict["fit"], legend_dict["data"], legend_dict["data2"]],
                    fontsize=14,
                    loc=legend_dict["loc"],
                )
            if (data_dict is None) & (A_dict is not None):
                ax.legend(
                    [(leg_fill[0], leg_line[0]), leg_pt2],
                    [legend_dict["fit"], legend_dict["data2"]],
                    fontsize=14,
                    loc=legend_dict["loc"],
                )

            pdf.savefig()
            plt.close()


def plot_icm_profiles(res_dir, file_ACCEPT, z):
    """
    Plot the cluster ICM profiles

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    file_ACCEPT: path to the ACCEPT result file if any
    z: the cluster redshift

    Returns
    _______
    Creates .pdf files in the *figures* folder of the *results*
    directory in res_dir showing the cluster ICM profiles

    """

    mer_dir = res_dir + "/results/"
    fig_dir = mer_dir + "figures/"
    cl_dir = mer_dir + "cluster/"
    icm_dir = mer_dir + "ICM/"
    mcmc_dir_ne = mer_dir + "MCMC_ne/"

    print(colored("Plotting ICM profiles...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_Mhse = fig_dir + "Mhse_profile.pdf"

    if os.path.exists(file_Mhse):
        print(colored("ICM profiles already plotted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        f_A = False
        if file_ACCEPT is not None:
            f_A = True
            A_res = ascii.read(file_ACCEPT)
            A_rmid = 1e3 * (A_res["Rin"] + A_res["Rout"]) / 2.0

        icm_file = icm_dir + "ICM_best_fits.npz"
        ICM = np.load(icm_file)

        saved_Tx_prof_file = cl_dir + "T_prof_fit.npz"
        saved_Tx_prof = np.load(saved_Tx_prof_file)
        theta = saved_Tx_prof["datax"]
        Tx_data = saved_Tx_prof["datay"]
        Tx_data_xerr = saved_Tx_prof["dataxerr"]
        Tx_data_yerr = saved_Tx_prof["datayerr"]

        bestfit_Sx_file = mcmc_dir_ne + "ne_best_fit.npz"
        bestfit_Sx = np.load(bestfit_Sx_file)
        Sx = bestfit_Sx["Sx"]
        Sx_erru = bestfit_Sx["Sx_erru"]
        Sx_errd = bestfit_Sx["Sx_errd"]

        saved_xsb_prof_file = mer_dir + "cl_bkg_sb_profile.npz"
        saved_xsb_prof = np.load(saved_xsb_prof_file)
        theta_arcmin = saved_xsb_prof["r"]
        Sx_data = saved_xsb_prof["xsb"]
        Sx_data_err = saved_xsb_prof["xsb_err"]
        bkg_prof = saved_xsb_prof["xsb_bkg"]
        XSB_float_to_N = saved_xsb_prof["f2n"]

        Sx_data_p = Sx_data * XSB_float_to_N
        Sx_data_err_p = Sx_data_err * XSB_float_to_N
        bkg_prof_p = bkg_prof * XSB_float_to_N

        Sx_p = Sx * XSB_float_to_N
        Sx_erru_p = Sx_erru * XSB_float_to_N
        Sx_errd_p = Sx_errd * XSB_float_to_N

        file_Sx = fig_dir + "Sx_profile.pdf"
        title_dict = {
            "xlab": "$\mathrm{Angular~distance~[arcmin]}$",
            "ylab": "$\mathrm{S_X~[counts / s / arcmin^{2}]}$",
        }
        frame_dict = {
            "logx": 1,
            "logy": 1,
            "xr": [0.9 * np.min(theta_arcmin), 1.1 * np.max(theta_arcmin)],
            "yr": [0.5 * Sx_data_p[-1], 1.1 * np.max(Sx_p + 2.0 * Sx_erru_p)],
        }
        data_dict = {
            "x": theta_arcmin,
            "y": Sx_data_p,
            "xerr": None,
            "yerr": Sx_data_err_p,
        }
        legend_dict = {
            "data": "$\mathrm{Data}$",
            "fit": "$\mathrm{Best~fit}$",
            "loc": 3,
        }
        profile_plot(
            file_Sx,
            theta_arcmin,
            Sx_p,
            Sx_erru_p,
            Sx_errd_p,
            title_dict,
            frame_dict,
            legend_dict,
            data_dict,
        )

        maxr = np.max(theta + 2.0 * Tx_data_xerr)

        wr = np.where((ICM["r"] > 10.0) & (ICM["r"] < maxr))

        file_ne = fig_dir + "ne_profile.pdf"
        title_dict = {
            "xlab": "$\mathrm{Radius~[kpc]}$",
            "ylab": "$\mathrm{n_e~[cm^{-3}]}$",
        }
        frame_dict = {
            "logx": 1,
            "logy": 1,
            "xr": [10, maxr],
            "yr": [
                0.95 * np.nanmin(ICM["ne"][wr]),
                1.05 * np.nanmax((ICM["ne"] + 2.0 * ICM["ne_erru"])[wr]),
            ],
        }
        if f_A:
            A_dict = {"x": A_rmid, "y": A_res["nelec"], "yerr": A_res["neerr"]}
            legend_dict = {
                "data2": "$\mathrm{ACCEPT~results}$",
                "fit": "$\mathrm{FR~results}$",
                "loc": 3,
            }
        else:
            A_dict, legend_dict = [None, None]
        data_dict = None
        profile_plot(
            file_ne,
            ICM["r"],
            ICM["ne"],
            ICM["ne_erru"],
            ICM["ne_errd"],
            title_dict,
            frame_dict,
            legend_dict,
            data_dict,
            A_dict,
        )

        icm_prop = np.load(icm_dir + "ICM_best_fits.npz")
        R500_icm = float(icm_prop["R500"])
        rho_prof = (
            const.m_p.to("g").value
            * (1.397 / 1.199)
            * ICM["ne"]
            / cosmo.critical_density(z).value
        )
        rho_prof_erru = (
            const.m_p.to("g").value
            * (1.397 / 1.199)
            * ICM["ne_erru"]
            / cosmo.critical_density(z).value
        )
        rho_prof_errd = (
            const.m_p.to("g").value
            * (1.397 / 1.199)
            * ICM["ne_errd"]
            / cosmo.critical_density(z).value
        )
        file_rho = fig_dir + "rho_profile.pdf"
        title_dict = {"xlab": "$\mathrm{R/R_{500}}$", "ylab": "$\\rho/\\rho_c$"}
        frame_dict = {
            "logx": 1,
            "logy": 1,
            "xr": [1e-2, maxr / R500_icm],
            "yr": [
                np.max([1.0, 0.95 * np.nanmin(rho_prof[wr])]),
                1.05 * np.nanmax((rho_prof + 2.0 * rho_prof_erru)[wr]),
            ],
        }
        A_dict, legend_dict = [None, None]
        data_dict = None
        profile_plot(
            file_rho,
            ICM["r"] / R500_icm,
            rho_prof,
            rho_prof_erru,
            rho_prof_errd,
            title_dict,
            frame_dict,
            legend_dict,
            data_dict,
            A_dict,
        )

        file_pe = fig_dir + "pe_profile.pdf"
        title_dict = {
            "xlab": "$\mathrm{Radius~[kpc]}$",
            "ylab": "$\mathrm{P_e~[keV\,cm^{-3}]}$",
        }
        frame_dict = {
            "logx": 1,
            "logy": 1,
            "xr": [10, maxr],
            "yr": [
                0.95 * np.nanmin(ICM["pe"][wr]),
                1.05 * np.nanmax((ICM["pe"] + 2.0 * ICM["pe_erru"])[wr]),
            ],
        }
        profile_plot(
            file_pe,
            ICM["r"],
            ICM["pe"],
            ICM["pe_erru"],
            ICM["pe_errd"],
            title_dict,
            frame_dict,
        )

        file_tcool = fig_dir + "tcool_profile.pdf"
        title_dict = {
            "xlab": "$\mathrm{Radius~[kpc]}$",
            "ylab": "$\mathrm{t_{cool}~[Gyr]}$",
        }
        frame_dict = {
            "logx": 1,
            "logy": 1,
            "xr": [10, maxr],
            "yr": [
                0.95 * np.nanmin(ICM["tcool"][wr]),
                1.05 * np.nanmax((ICM["tcool"] + 2.0 * ICM["tcool_erru"])[wr]),
            ],
        }
        profile_plot(
            file_tcool,
            ICM["r"],
            ICM["tcool"],
            ICM["tcool_erru"],
            ICM["tcool_errd"],
            title_dict,
            frame_dict,
        )

        file_te = fig_dir + "te_profile.pdf"
        title_dict = {
            "xlab": "$\mathrm{Radius~[kpc]}$",
            "ylab": "$\mathrm{k_BT_e~[keV]}$",
        }
        frame_dict = {
            "logx": 1,
            "logy": 0,
            "xr": [10, maxr],
            "yr": [0.0, np.nanmax([20.0, np.nanmax(Tx_data + Tx_data_yerr)])],
        }
        data_dict = {
            "x": theta,
            "y": Tx_data,
            "xerr": Tx_data_xerr,
            "yerr": Tx_data_yerr,
        }
        if f_A:
            A_dict = {"x": A_rmid, "y": A_res["Tx"], "yerr": A_res["Txerr"]}
            legend_dict = {
                "data": "$\mathrm{Spectroscopic~data}$",
                "data2": "$\mathrm{ACCEPT~values}$",
                "fit": "$\mathrm{FR~results}$",
                "loc": 2,
            }
        else:
            A_dict = None
            legend_dict = {
                "data": "$\mathrm{Spectroscopic~data}$",
                "fit": "$\mathrm{Best~fit}$",
                "loc": 2,
            }
        profile_plot(
            file_te,
            ICM["r"],
            ICM["te"],
            ICM["te_erru"],
            ICM["te_errd"],
            title_dict,
            frame_dict,
            legend_dict,
            data_dict,
            A_dict,
        )

        file_ke = fig_dir + "ke_profile.pdf"
        title_dict = {
            "xlab": "$\mathrm{Radius~[kpc]}$",
            "ylab": "$\mathrm{K_e~[keV\,cm^2]}$",
        }
        frame_dict = {
            "logx": 1,
            "logy": 1,
            "xr": [10, maxr],
            "yr": [5.0, 1.05 * np.nanmax((ICM["ke"] + 2.0 * ICM["ke_erru"])[wr])],
        }
        if f_A:
            A_dict = {"x": A_rmid, "y": A_res["Kitpl"], "yerr": A_res["Kerr"]}
            legend_dict = {
                "data2": "$\mathrm{ACCEPT~results}$",
                "fit": "$\mathrm{FR~results}$",
                "loc": 4,
            }
        else:
            A_dict, legend_dict = [None, None]
        data_dict = None
        profile_plot(
            file_ke,
            ICM["r"],
            ICM["ke"],
            ICM["ke_erru"],
            ICM["ke_errd"],
            title_dict,
            frame_dict,
            legend_dict,
            data_dict,
            A_dict,
        )

        title_dict = {
            "xlab": "$\mathrm{Radius~[kpc]}$",
            "ylab": "$\mathrm{M_{HSE}~[M_{\odot}]}$",
        }
        frame_dict = {
            "logx": 1,
            "logy": 1,
            "xr": [10, maxr],
            "yr": [1e12, 1.05 * np.nanmax((ICM["Mhse"] + 2.0 * ICM["Mhse_erru"])[wr])],
        }
        profile_plot(
            file_Mhse,
            ICM["r"],
            ICM["Mhse"],
            ICM["Mhse_erru"],
            ICM["Mhse_errd"],
            title_dict,
            frame_dict,
        )


def plot_2D_posteriors(res_dir, N_ann):
    """
    Creates the MCMC triangle plots

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    N_ann: number of annuli for the spectrum extraction

    Returns
    _______
    Creates .pdf files in the *figures* folder of the *results*
    directory in res_dir showing the triangle plots

    """

    mer_dir = res_dir + "/results/"
    mcmc_dir_ne = mer_dir + "MCMC_ne/"
    mcmc_dir_pe = mer_dir + "MCMC_pe/"
    fig_dir = mer_dir + "figures/"

    print(colored("Plotting 2D distributions...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    if os.path.exists(fig_dir + "MCMC_corner_plot_ne.pdf"):
        print(colored("Corner plot already plotted", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        labels = [
            r"${n_{e0}\;\; [\mathrm{cm^{-3}}]}$",
            r"${r_{c}\;\; [\mathrm{kpc}]}$",
            r"${\alpha}$",
            r"${\beta}$",
            r"${r_{s}\;\; [\mathrm{kpc}]}$",
            r"${\epsilon}$",
            r"${\mathrm{Bkg_scale}}$",
        ]

        file_samples = mcmc_dir_ne + "MCMC_chains_clean.npz"
        results = np.load(file_samples)
        samples = results["samp"]

        file_bfit = mcmc_dir_ne + "best_fit_params.npy"
        best_fit_param = np.load(file_bfit)
        ndim = best_fit_param.size

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = MCSamples(samples=samples)
            samples.updateSettings({"contours": [0.68, 0.95]})
            fullpath = fig_dir + "MCMC_corner_plot_ne.pdf"
            with PdfPages(fullpath) as pdf:
                g = plots.getSubplotPlotter()
                g.settings.colormap = "gist_heat"
                g.triangle_plot(
                    [samples],
                    shaded=True,
                    contour_lws=[2],
                    xmarkers=best_fit_param,
                    ymarkers=best_fit_param,
                    marker_args={
                        "color": "#13428F",
                        "lw": 2,
                        "ls": (0, (3, 2, 1, 2)),
                        "dash_capstyle": "round",
                    },
                )
                for i in range(ndim):
                    g.subplots[ndim - 1, i].set_xlabel(labels[i], fontsize=14)
                    if i > 0:
                        g.subplots[i, 0].set_ylabel(labels[i], fontsize=14)
                        yax = g.subplots[i, 0].get_yaxis()
                        yax.set_label_coords(-0.2, 0.5)
                pdf.savefig()
                plt.close()

        if N_ann > 2:
            labels = [
                r"${P_{0}\;\; [\mathrm{keV\,cm^{-3}}]}$",
                r"${r_{p}\;\; [\mathrm{kpc}]}$",
                r"$a$",
                r"$b$",
                r"$c$",
            ]

            file_samples = mcmc_dir_pe + "MCMC_chains_clean.npz"
            results = np.load(file_samples)
            samples = results["samp"]

            file_bfit = mcmc_dir_pe + "best_fit_params.npy"
            best_fit_param = np.load(file_bfit)
            ndim = best_fit_param.size

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = MCSamples(samples=samples)
                samples.updateSettings({"contours": [0.68, 0.95]})
                fullpath = fig_dir + "MCMC_corner_plot_pe.pdf"
                with PdfPages(fullpath) as pdf:
                    g = plots.getSubplotPlotter()
                    g.settings.colormap = "gist_heat"
                    g.triangle_plot(
                        [samples],
                        shaded=True,
                        contour_lws=[2],
                        xmarkers=best_fit_param,
                        ymarkers=best_fit_param,
                        marker_args={
                            "color": "#13428F",
                            "lw": 2,
                            "ls": (0, (3, 2, 1, 2)),
                            "dash_capstyle": "round",
                        },
                    )
                    for i in range(ndim):
                        g.subplots[ndim - 1, i].set_xlabel(labels[i], fontsize=14)
                        if i > 0:
                            g.subplots[i, 0].set_ylabel(labels[i], fontsize=14)
                            yax = g.subplots[i, 0].get_yaxis()
                            yax.set_label_coords(-0.2, 0.5)
                    pdf.savefig()
                    plt.close()


def adaptive_map(res_dir, z, R500):
    """
    Produces a map of the cluster with
    adaptive smoothing

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    z: the cluster redshift
    R500: the cluster R500 radius in kpc

    Returns
    _______
    Creates a .pdf file in the *figures* folder of the *results*
    directory in res_dir showing the cluster map

    """

    mer_dir = res_dir + "/results/"
    fig_dir = mer_dir + "figures/"

    print(colored("Making adaptively smoothed image...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    map_file_pdf = fig_dir + "Adapt_smooth_img.pdf"

    if os.path.exists(map_file_pdf):
        print(colored("Image already produced", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        map_file_large = mer_dir + "wide_broad_thresh_nopts.img"
        map_file_small = mer_dir + "small_broad_thresh_nopts.img"

        depro_center = ascii.read(mer_dir + "peak_cent_pos.txt")
        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        theta_500 = (((R500 * u.kpc).value / d_a) * u.rad).to("arcmin").value
        crop_fits_to_FoV(
            map_file_large,
            map_file_small,
            depro_center["X_Y_cent_coord"],
            3.0 * theta_500,
        )

        map_file_out = mer_dir + "Adapt_smooth_img.fits"

        sp.call(["bash", "shell/adap_smooth.sh", map_file_small, map_file_out, mer_dir])

        ruler_l = ((100.0 / d_a) * u.rad).to("arcsec").value

        map_display(
            map_file_out,
            map_file_pdf,
            c_title="$\mathrm{Arbitrary~unit}$",
            centroid=depro_center["X_Y_cent_coord"],
            peak=depro_center["X_Y_peak_coord"],
            ruler=ruler_l,
            ruler_label="$\mathrm{100~kpc}$",
        )


def compute_Aphot(res_dir, z, R500):
    """
    Computes the photon asymmetry  morphological indicator
    following Nurgaliev et al. 2013
    (https://iopscience.iop.org/article/10.1088/0004-637X/779/2/112/pdf)

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    z: the cluster redshift
    R500: the cluster R500 radius in kpc

    Returns
    _______
    Creates a .txt file called Aphot.txt containing
    the value of the photon asymmetry

    """

    mer_dir = res_dir + "/results/"

    print(colored("Computing photon asymmetry...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    Aphot_file = mer_dir + "Aphot_morpho.npy"

    if os.path.exists(Aphot_file):
        print(colored("Photon asymmetry already computed", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        map_file_small = mer_dir + "small_broad_thresh_nopts.img"
        hdu = fits.open(map_file_small)
        header = hdu[0].header
        map_small = hdu[0].data

        pix_reso = header["CDELT2"]
        sig_kernel = (
            (((40.0 * u.kpc) / cosmo.angular_diameter_distance(z).to("kpc")) * u.rad)
            .to("deg")
            .value
        )
        sig_kernel_pix = sig_kernel / pix_reso

        map_conv = scipy.ndimage.filters.gaussian_filter(map_small, sig_kernel_pix)
        npix = map_conv.shape[1]

        Xpeak_img = map_conv.argmax() % map_conv.shape[1]
        Ypeak_img = map_conv.argmax() / map_conv.shape[1]

        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        theta_500 = (((R500 * u.kpc).value / d_a) * u.rad).to("deg").value
        theta_500_pix = theta_500 / pix_reso

        r_x = np.arange(0, int(npix / 2) + 1)
        r_xy = np.hypot(
            *np.meshgrid(
                np.concatenate((-np.flip(r_x[1:]), r_x)) - (Xpeak_img - int(npix / 2)),
                np.concatenate((-np.flip(r_x[1:]), r_x)) - (Ypeak_img - int(npix / 2)),
            )
        )
        if not npix % 2:
            r_xy = r_xy[:-1, :-1]

        ann_tab = np.array([0.05, 0.12, 0.2, 0.30, 1]) * theta_500_pix

        wbkg = np.where((r_xy >= 1.1 * theta_500_pix) & (r_xy < 1.3 * theta_500_pix))
        area_bkg = (
            np.pi * (1.3 * theta_500_pix) ** 2 - np.pi * (1.1 * theta_500_pix) ** 2
        )
        N_bkg = np.sum(map_small[wbkg])
        rate_bkg = N_bkg / area_bkg

        tab_C_k = []
        tab_dNC_k = []

        ang_tab = np.linspace(0.0, 2.0 * np.pi, 10)

        for i in range(ann_tab.size - 1):

            U_N2_tab = np.zeros(10)

            for k, ang in enumerate(ang_tab):
                w = np.where((r_xy >= ann_tab[i]) & (r_xy < ann_tab[i + 1]))
                tab_counts = []
                tab_phi = []
                for j in range(w[0].size):
                    tab_counts.append(map_small[w[0][j], w[1][j]])
                    Dx = w[1][j] - Xpeak_img
                    Dy = w[0][j] - Ypeak_img
                    if (Dx >= 0) & (Dy >= 0):
                        phi = np.arctan(Dy / Dx)
                    elif (Dx >= 0) & (Dy < 0):
                        phi = 2.0 * np.pi + np.arctan(Dy / Dx)
                    else:
                        phi = np.arctan(Dy / Dx) + np.pi

                    if (phi + ang) < (2.0 * np.pi):
                        tab_phi.append(phi + ang)
                    else:
                        tab_phi.append(phi + ang - 2.0 * np.pi)

                tab_counts = np.array(tab_counts)
                tab_phi = np.array(tab_phi)
                sort_ind = np.argsort(tab_phi)
                N = np.sum(map_small[w])
                cumul_F = np.cumsum(tab_counts[sort_ind]) / N
                cumul_G = np.linspace(0, 1, cumul_F.size)

                U_N2_tab[k] = N * np.trapz((cumul_F - cumul_G) ** 2, cumul_G)

            U_N2 = np.min(U_N2_tab)
            area_ann = np.pi * ann_tab[i + 1] ** 2 - np.pi * ann_tab[i] ** 2
            B = rate_bkg * area_ann
            C = N - B
            if C > 0:
                d_N_C = (N / (C ** 2)) * (U_N2 - 1.0 / 12.0)
            else:
                C = 0
                d_N_C = 0

            tab_C_k.append(C)
            tab_dNC_k.append(d_N_C)

        tab_C_k = np.array(tab_C_k)
        tab_dNC_k = np.array(tab_dNC_k)
        Aphot = 100.0 * np.sum(tab_C_k * tab_dNC_k) / np.sum(tab_C_k)

        np.save(Aphot_file, Aphot)


def cluster_id_card(res_dir, source_name, z):
    """
    Creates the cluster ID card showing the
    name, redshift, mass, X-ray peak position, etc.

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    source_name: the cluster name
    z: the cluster redshift

    Returns
    _______
    Creates a .txt file in the *figures* folder of the *results*
    directory in res_dir containing the cluster ID card

    """

    mer_dir = res_dir + "/results/"
    icm_dir = mer_dir + "ICM/"
    fig_dir = mer_dir + "figures/"

    print(colored("Making cluster ID card...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    depro_center = ascii.read(mer_dir + "peak_cent_pos.txt")
    icm_prop = np.load(icm_dir + "ICM_best_fits.npz")

    Aphot_file = mer_dir + "Aphot_morpho.npy"
    Aphot = np.float(np.load(Aphot_file))

    cl_id = open(fig_dir + "cluster_ID_card.txt", "w")

    cl_id.write("#################################################\n")
    cl_id.write("Cluster Name: " + source_name + "\n")
    cl_id.write("#################################################\n")
    cl_id.write("\n")
    cl_id.write("z = " + "{:.2f}".format(z) + "\n")
    cl_id.write(
        "M500 = ("
        + "{:.2f}".format(float(icm_prop["MYx"]) * 1e-14)
        + " +/- "
        + "{:.2f}".format(float(icm_prop["MYx_err"]) * 1e-14)
        + ") x 10^14 Msun"
        + "\n"
    )
    cl_id.write("\n")
    cl_id.write("#################################################\n")
    cl_id.write("\n")
    cl_id.write(
        "X-ray centroid: R.A. =  "
        + "{:.4f}".format(depro_center["X_Y_cent_coord"][0])
        + " deg  ||  Dec. = "
        + "{:.4f}".format(depro_center["X_Y_cent_coord"][1])
        + " deg"
        + "\n"
    )
    cl_id.write(
        "X-ray peak:     R.A. =  "
        + "{:.4f}".format(depro_center["X_Y_peak_coord"][0])
        + " deg  ||  Dec. = "
        + "{:.4f}".format(depro_center["X_Y_peak_coord"][1])
        + " deg"
        + "\n"
    )
    cl_id.write("\n")
    cl_id.write("Aphot = " + "{:.3f}".format(Aphot))
    cl_id.write("\n")
    cl_id.write("#################################################\n")
    cl_id.close()

    with open(fig_dir + "cluster_ID_card.txt") as f:
        content = f.readlines()

    for i in range(len(content)):
        if len(content[i]) > 1:
            print(content[i][:-1])
