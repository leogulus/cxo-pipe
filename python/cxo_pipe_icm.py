from termcolor import colored
import warnings
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import astropy.units as u
from scipy.special import gammaln
import subprocess as sp
import os
import emcee
from multiprocessing import Pool
from contextlib import closing
from tqdm import tqdm
import astropy.constants as const
from pydl.pydlutils.cooling import read_ds_cooling
from astropy.io import fits
import time
import logging
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import matplotlib as mpl
from sherpa.astro import ui

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
mpl.rcParams["text.latex.preamble"] = r"\boldmath"
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

cosmo = FlatLambdaCDM(70.0, 0.3, Tcmb0=2.7255)


def extrap(x, xp, yp):
    """
    np.interp function with linear extrapolation.
    """
    y = np.interp(x, xp, yp)
    y[x < xp[0]] = yp[0] + (x[x < xp[0]] - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1])
    y[x > xp[-1]] = yp[-1] + (x[x > xp[-1]] - xp[-1]) * (yp[-1] - yp[-2]) / (
        xp[-1] - xp[-2]
    )

    return y


def mad(a, axis=None):
    """
    Compute *Median Absolute Deviation* of an array along given axis.
    """

    # Median along given axis, but *keeping* the reduced axis so that
    # result can still broadcast against a.
    med = np.nanmedian(a, axis=axis, keepdims=True)
    mad = np.nanmedian(np.absolute(a - med), axis=axis)  # MAD along given axis

    return mad


def VPM_model(r, param):

    """
    Computes the density profile based on a
    Vikhlinin parametric model (Vikhlinin+2006)

    Parameters
    __________
    r: array containing the radial range considered in kpc
    param: the model parameters, n0,rc,alpha,beta,rs,epsilon

    Returns
    _______
    model: the ICM density profile of the cluster
    across the considered radial range

    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = param[0] * (
            ((r / param[1]) ** (-param[2] / 2.0))
            / (
                (1 + (r / param[1]) ** 2) ** (3 * param[3] / 2.0 - param[2] / 4.0)
                * (1.0 + (r / param[4]) ** 3) ** (param[5] / 6)
            )
        )

    return model


def gNFW_model(r, param):
    """
    Computes the pressure profile based on a
    generalized Navarro-Frenk-White model (Nagai+2007))

    Parameter
    ----------
    r: array containing the radial range considered in kpc
    param: the model parameters, P0,rp,a,b,c

    Returns
    _______
    model: the ICM pressure profile of the cluster
    across the considered radial range

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf = r / param[1]
        model = param[0] / (
            ((rf) ** param[4])
            * ((1 + (rf) ** param[2])) ** ((param[3] - param[4]) / param[2])
        )

    return model


def init_integ_maps(z, R500):

    """
    Initialize the 2D radius map and the
    line of sight step map to be used to compute
    the emission measure profile

    Parameters
    __________
    z: the cluster redshift
    R500: the cluster R500 radius in kpc

    Returns
    _______
    Rproj: a 1D array that gives the projected radius
    considered to build the 2D array los_step_map
    r_map: a 2D array used to compute the ICM
    density distribution
    los_step_map: a 2D array used to compute the
    integral of the ICM density distribution
    along the line of sight

    """

    # Define general parameters
    los_Nstep = 100  # Number of los step
    delta_Rproj = 1.0  # Projected radius steps [arcsec]
    delta_min_los = 1.0  # Min los step [kpc]
    d_a = cosmo.angular_diameter_distance(z).to("kpc").value
    # Define integration parameters
    r_out = 6.0 * R500  # Above this radius we consider out of cluster
    rproj_out = (
        2.0 * R500
    )  # Maximum projected radius at which the corresponding projected profile is calculated
    # Line-of-Sight grid
    los_min = delta_min_los  # kpc (take 1 kpc for min step)
    los_max = r_out  # kpc
    los_bin = np.logspace(np.log10(los_min), np.log10(los_max), los_Nstep)
    los_bin = np.append(0, los_bin)
    los_step = los_bin - np.roll(los_bin, 1)  # Bin size
    los_cbin = (los_bin + np.roll(los_bin, 1)) / 2.0  # Center of bins
    los_step = los_step[1:]  # first bin does not really exist
    los_cbin = los_cbin[1:]  # first bin does not really exist
    # Projected radius grid
    rproj_step = (
        delta_Rproj * np.pi / 180.0 / 3600.0
    ) * d_a  # kpc (take 2 arcsec step)
    rproj_max = rproj_out  # kpc
    rproj_Nstep = int(rproj_max / rproj_step)  # Number of bins along L.o.S
    remainder_4_div = rproj_Nstep % 4
    if remainder_4_div != 0:
        rproj_Nstep += 4 - remainder_4_div
    Rproj = np.linspace(
        rproj_step, rproj_out, num=rproj_Nstep
    )  # Vector going up to rproj_out with rp_step steps
    # Compute the radius grid
    los_cbin_map = (
        np.outer(los_cbin, np.ones(rproj_Nstep))
    ).T  # Line of sight bin central value map
    los_step_map_kpc = (
        np.outer(los_step, np.ones(rproj_Nstep))
    ).T  # Line of sight bin size map
    los_step_map = (los_step_map_kpc * u.kpc).to("cm").value
    Rproj_map = np.outer(Rproj, np.ones(los_cbin.size))  # Projected radius map
    r_map = np.sqrt(
        Rproj_map ** 2 + los_cbin_map ** 2
    )  # Physical radius from the center map

    return [Rproj, r_map, los_step_map]


def XSB_model(
    param, theta, bkg_prof, XSB_float_to_N, conv_fact, Rproj, r_map, los_step_map, R500
):

    """
    Computes the X-ray surface brightness profile from an
    ICM density model, a set of conversion coefficients
    from EM to XSB, and a background model

    Parameters
    __________
    theta: projected radius of the surface brightness profile
    param: the model parameters, n0,rc,alpha,beta,rs,epsilon
    Rproj: a 1D array that gives the projected radius
    considered to build the 2D array los_step_map
    conv_fact: the conversion coefficients from EM to XSB
    bkg_prof: the number of counts due to background in each annulus
    XSB_float_to_N: conversion from surface brightness to number count
    R500: the cluster R500 radius in kpc

    Returns
    _______
    Sx_prof: the X-ray surface brightness profile model to be
    compared with Chandra data

    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R_max = 6.0 * R500
        ne_map = VPM_model(r_map, param)
        integrant = ((ne_map ** 2) / 1.199) * los_step_map
        lim_out = np.where(r_map > R_max)
        nlim_out = np.asarray(lim_out).size
        if nlim_out > 0:
            integrant[lim_out] = 0

        EM_prof_all = 2.0 * integrant.sum(axis=1)
        EM_prof_pow = extrap(np.log10(theta), np.log10(Rproj), np.log10(EM_prof_all))
        EM_prof = 10 ** EM_prof_pow

        conv_fact_i = conv_fact[:, int(np.random.uniform(0, conv_fact.shape[1]))]

        Sx_prof_cl = EM_prof / conv_fact_i
        Sx_prof = Sx_prof_cl / XSB_float_to_N + param[-1] * bkg_prof

    return Sx_prof


def ln_likelihood_ne(
    param,
    theta,
    Sx_data,
    bkg_prof,
    XSB_float_to_N,
    conv_fact,
    Rproj,
    r_map,
    los_step_map,
    R500,
):

    """
    Computes the log-likelihood function used to fit
    the surface brightness profile

    Parameters
    __________
    theta: projected radius of the surface brightness profile
    param: the model parameters, n0,rc,alpha,beta,rs,epsilon
    Rproj: a 1D array that gives the projected radius
    considered to build the 2D array los_step_map
    conv_fact: the conversion coefficients from EM to XSB
    bkg_prof: the number of counts due to background in each annulus
    XSB_float_to_N: conversion from surface brightness to number count
    R500: the cluster R500 radius in kpc

    Returns
    _______
    lnlike: the value of the log-likelihood for the
    considered set of model parameters

    """

    Sx_model = XSB_model(
        param,
        theta,
        bkg_prof,
        XSB_float_to_N,
        conv_fact,
        Rproj,
        r_map,
        los_step_map,
        R500,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lnlike_tab = Sx_data * np.log(Sx_model) - Sx_model - gammaln(Sx_data + 1)
        lnlike = np.sum(lnlike_tab)

    if not np.isfinite(lnlike):
        return -np.inf
    return lnlike


def ln_likelihood_pe(
    param, theta, Tx_data, Tx_data_err, ne_fit, ne_fit_erru, ne_fit_errd
):

    """
    Computes the log-likelihood function used to fit
    the spectroscopic temperature profile

    Parameters
    __________
    param: the model parameters, P0,rp,a,b,c
    theta: projected radius of the temperature profile
    Tx_data: the spectroscopic temperature profile
    Tx_data_err: the error bars associated with Tx_data
    ne_fit: the best-fit ICM density model
    ne_fit_err: the error bars associated with ne_model

    Returns
    _______
    lnlike: the value of the log-likelihood for the
    considered set of model parameters

    """

    Pe_model = gNFW_model(theta, param)
    if np.random.uniform(-1, 1) < 0:
        ne_model = ne_fit - np.abs(np.random.normal()) * ne_fit_errd
    else:
        ne_model = ne_fit + np.abs(np.random.normal()) * ne_fit_erru

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Te_model = Pe_model / ne_model
        lnlike_tab = -0.5 * ((Tx_data - Te_model) / Tx_data_err) ** 2
        lnlike = np.sum(lnlike_tab)

    if not np.isfinite(lnlike):
        return -np.inf
    return lnlike


def ln_prior_ne(param):
    """
    Computes the log-prior distribution used to fit
    the surface brightness profile

    Parameters
    __________
    param: the model parameters, n0,rc,alpha,beta,rs,epsilon

    Returns
    _______
    The value of the log-prior for the
    considered set of model parameters

    """

    n0, rc, alpha, beta, rs, epsilon, bkg_scale = param

    check_term = 0.0
    if n0 > 0 and rc > 0 and rc < 1e3 and rs > 0 and rs < 5e3 and alpha > 0:
        check_term += 1

    bkg_err = 0.5  # Assume the background is know at better than 50%
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bkg_prior = -0.5 * ((bkg_scale - 1.0) / bkg_err) ** 2

    if check_term < 1:
        return -np.inf
    else:
        return bkg_prior


def ln_prior_pe(param, r_test, VPM_param):
    """
    Computes the log-prior distribution used to fit
    the spectroscopic temperature profile

    Parameters
    __________
    param: the model parameters, P0,rp,a,b,c
    r_test: a 3D radius array used to compute the model (in kpc)
    VPM_param: the best-fit VPM parameters of the density profile

    Returns
    _______
    The value of the log-prior for the
    considered set of model parameters

    """

    P0, rp, a, b, c = param

    check_term = 0.0
    if (
        P0 > 0
        and rp > 0
        and rp < 5e3
        and a > 0
        and a < 5
        and b > 3
        and b < 20
        and c > 0.0
        and c < 1.1
    ):
        check_term += 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Pe = gNFW_model(r_test, param)
        ne = VPM_model(r_test, VPM_param[:-1])

        Te_prof = Pe / ne
        d1_Te_prof = (np.gradient(Te_prof, r_test))[0:-1]
        d2_Te_prof = (np.gradient(d1_Te_prof, r_test[0:-1]))[0:-1]

        Ke_prof = Pe / ne ** (5.0 / 3.0)
        d1_Ke_prof = (np.gradient(Ke_prof, r_test))[0:-1]

        dpdr = (
            -P0
            * (r_test / rp) ** (-c)
            * (1 + (r_test / rp) ** a) ** (-(a + b - c) / a)
            * (b * (r_test / rp) ** a + c)
            / r_test
        )
        MHSE_r = -1.0 * r_test ** 2 * dpdr / ne
        d1_MHSE_r = (np.gradient(MHSE_r, r_test))[0:-1]

    if all(d1_Ke >= 0 for d1_Ke in d1_Ke_prof) & all(d1_M >= 0 for d1_M in d1_MHSE_r):
        check_term += 1

    if check_term < 2:
        return -np.inf
    else:
        return 0.0


def ln_posterior_ne(
    param,
    theta,
    Sx_data,
    bkg_prof,
    XSB_float_to_N,
    conv_fact,
    Rproj,
    r_map,
    los_step_map,
    R500,
):
    """
    Computes the log-posterior distribution used to fit
    the surface brightness profile

    Parameters
    __________
    theta: projected radius of the surface brightness profile
    param: the model parameters, n0,rc,alpha,beta,rs,epsilon
    Rproj: a 1D array that gives the projected radius
    considered to build the 2D array los_step_map
    conv_fact: the conversion coefficients from EM to XSB
    bkg_prof: the number of counts due to background in each annulus
    XSB_float_to_N: conversion from surface brightness to number count
    R500: the cluster R500 radius in kpc

    Returns
    _______
    The value of the log-posterior for the
    considered set of model parameters

    """

    lp = ln_prior_ne(param)
    if not np.isfinite(lp):
        return -np.inf
    else:
        ll = ln_likelihood_ne(
            param,
            theta,
            Sx_data,
            bkg_prof,
            XSB_float_to_N,
            conv_fact,
            Rproj,
            r_map,
            los_step_map,
            R500,
        )
        return lp + ll


def ln_posterior_pe(
    param,
    theta,
    Tx_data,
    Tx_data_err,
    ne_fit,
    ne_fit_erru,
    ne_fit_errd,
    r_test,
    VPM_param,
):
    """
    Computes the log-posterior distribution used to fit
    the spectroscopic temperature profile

    Parameters
    __________
    param: the model parameters, P0,rp,a,b,c
    theta: projected radius of the temperature profile
    Tx_data: the spectroscopic temperature profile
    Tx_data_err: the error bars associated with Tx_data
    ne_fit: the best-fit ICM density model
    ne_fit_err: the error bars associated with ne_model
    r_test: a 3D radius array used to compute the model (in kpc)
    VPM_param: the best-fit VPM parameters of the density profile

    Returns
    _______
    The value of the log-posterior for the
    considered set of model parameters

    """

    lp = ln_prior_pe(param, r_test, VPM_param)
    if not np.isfinite(lp):
        return -np.inf
    else:
        ll = ln_likelihood_pe(
            param, theta, Tx_data, Tx_data_err, ne_fit, ne_fit_erru, ne_fit_errd
        )
        return lp + ll


def mcmc_sampler_ne(pos, ndim, nwalkers, nsteps, args_lnprob):

    """
    Performs a MCMC sampling of the parameter
    space based on the likelihood function defined
    to fit the surface brightness profile

    Parameters
    __________
    pos: the initial guess of the parameters for each chain
    ndim: the number of model parameters
    nwalkers: the number of chains in the MCMC
    nsteps: the number of steps
    args_lnprob: the arguments of the posterior distribution

    Returns
    _______
    sampler: the structure containing the chains of
    parameters and the posterior values
    """

    with closing(Pool()) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, ln_posterior_ne, a=2.0, args=args_lnprob, pool=pool
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler.run_mcmc(pos, nsteps, progress=True)

        pool.terminate()

    return sampler


def mcmc_sampler_pe(pos, ndim, nwalkers, nsteps, args_lnprob):

    """
    Performs a MCMC sampling of the parameter
    space based on the likelihood function defined
    to fit the spectroscopic temperature profile

    Parameters
    __________
    pos: the initial guess of the parameters for each chain
    ndim: the number of model parameters
    nwalkers: the number of chains in the MCMC
    nsteps: the number of steps
    args_lnprob: the arguments of the posterior distribution

    Returns
    _______
    sampler: the structure containing the chains of
    parameters and the posterior values
    """

    with closing(Pool()) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, ln_posterior_pe, a=2.0, args=args_lnprob, pool=pool
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler.run_mcmc(pos, nsteps, progress=True)

        pool.terminate()

    return sampler


def mcmc_ne(res_dir, Rproj, r_map, los_step_map, z, R500):
    """
    Run the MCMC analysis to find the best-fit model
    of the ICM density profile

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    Rproj: a 1D array that gives the projected radius
    considered to build the 2D array los_step_map
    r_map: a 2D array used to compute the ICM
    density distribution
    los_step_map: a 2D array used to compute the
    integral of the ICM density distribution
    along the line of sight
    z: the cluster redshift
    R500: the cluster R500 radius in kpc

    Returns
    _______
    Create a .npz file in the *MCMC_ne* folder of the *results*
    directory in res_dir containing the chains of parameters
    and likelihood values

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    mcmc_dir = mer_dir + "MCMC_ne/"

    if not os.path.exists(mcmc_dir):
        sp.call("mkdir " + mcmc_dir, shell=True)

    print(colored("Runnning Sx profile MCMC analysis...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_chains = mcmc_dir + "MCMC_chains.npz"

    if os.path.exists(file_chains):
        print(colored("MCMC analysis already done", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        saved_xsb_prof_file = mer_dir + "cl_bkg_sb_profile.npz"
        saved_xsb_prof = np.load(saved_xsb_prof_file)
        theta_arcmin = saved_xsb_prof["r"]
        Sx_data = saved_xsb_prof["xsb"]
        bkg_prof = saved_xsb_prof["xsb_bkg"]
        XSB_float_to_N = saved_xsb_prof["f2n"]

        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        theta = d_a * (theta_arcmin * u.arcmin).to("radian").value

        file_conv_tab = cl_dir + "conv_tab.npy"
        conv_fact = np.load(file_conv_tab)

        param_ini = np.asarray([2e-2, 150.0, 0.1, 1.2, 1000.0, 3.0, 1.0])

        ndim, nwalkers, nsteps = param_ini.size, 400, 250

        n0 = np.random.uniform(0.0, 5.0 * param_ini[0], nwalkers)
        rc = np.random.uniform(0.0, 1000.0, nwalkers)
        alpha = np.random.uniform(-5 * param_ini[2], 5.0 * param_ini[2], nwalkers)
        beta = np.random.uniform(-5.0 * param_ini[3], 5.0 * param_ini[3], nwalkers)
        rs = np.random.uniform(0.0, 5000.0, nwalkers)
        epsilon = np.random.uniform(0.0, 5.0 * param_ini[5], nwalkers)
        bkg_scale = np.random.uniform(0.5, 1.5, nwalkers)
        pos_tab = np.vstack((n0, rc, alpha, beta, rs, epsilon, bkg_scale)).T
        pos_tab[-1, :] = param_ini
        pos = pos_tab.tolist()

        args_lnprob = [
            theta,
            Sx_data,
            bkg_prof,
            XSB_float_to_N,
            conv_fact,
            Rproj,
            r_map,
            los_step_map,
            R500,
        ]
        sampler = mcmc_sampler_ne(pos, ndim, nwalkers, nsteps, args_lnprob)

        chi2_chains = -2.0 * sampler.lnprobability
        min_chi2_chains = np.argsort(chi2_chains[:, -1])
        param_chains = sampler.chain

        ndim, nwalkers, nsteps = param_ini.size, 15, 10000
        pos = param_chains[min_chi2_chains[0:nwalkers], -1, :].tolist()
        sampler = mcmc_sampler_ne(pos, ndim, nwalkers, nsteps, args_lnprob)

        np.savez(file_chains, param=sampler.chain, lnprob=sampler.lnprobability)


def mcmc_pe(res_dir):
    """
    Run the MCMC analysis to find the best-fit model
    of the ICM pressure profile

    Parameters
    __________
    res_dir: the result directory named after the cluster name

    Returns
    _______
    Create a .npz file in the *MCMC_pe* folder of the *results*
    directory in res_dir containing the chains of parameters
    and likelihood values

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    mcmc_dir_ne = mer_dir + "MCMC_ne/"
    mcmc_dir = mer_dir + "MCMC_pe/"

    if not os.path.exists(mcmc_dir):
        sp.call("mkdir " + mcmc_dir, shell=True)

    print(
        colored("Runnning temperature profile MCMC analysis...", "blue", None, ["bold"])
    )
    print("------------------------------------------------------------")

    file_chains = mcmc_dir + "MCMC_chains.npz"

    if os.path.exists(file_chains):
        print(colored("MCMC analysis already done", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        saved_Tx_prof_file = cl_dir + "T_prof_fit.npz"
        saved_Tx_prof = np.load(saved_Tx_prof_file)
        theta = saved_Tx_prof["datax"]
        Tx_data = saved_Tx_prof["datay"]
        Tx_data_err = saved_Tx_prof["datayerr"]

        bestfit_ne_file = mcmc_dir_ne + "ne_best_fit.npz"
        bestfit_ne = np.load(bestfit_ne_file)
        r_model = bestfit_ne["r"]
        ne_model = bestfit_ne["ne"]
        ne_model_erru = bestfit_ne["ne_erru"]
        ne_model_errd = bestfit_ne["ne_errd"]

        ne_int_pow = np.interp(np.log10(theta), np.log10(r_model), np.log10(ne_model))
        ne_fit = 10.0 ** ne_int_pow

        ne_erru_int_pow = np.interp(
            np.log10(theta), np.log10(r_model), np.log10(ne_model_erru)
        )
        ne_fit_erru = 10.0 ** ne_erru_int_pow

        ne_errd_int_pow = np.interp(
            np.log10(theta), np.log10(r_model), np.log10(ne_model_errd)
        )
        ne_fit_errd = 10.0 ** ne_errd_int_pow

        r_test = np.logspace(1.0, 3.0, 100)

        VPM_param_file = mcmc_dir_ne + "best_fit_params.npy"
        VPM_param = np.load(VPM_param_file)

        param_ini = np.asarray([8.4e-2, 850.0, 1.0510, 5.4905, 0.3081])

        ndim, nwalkers, nsteps = param_ini.size, 400, 250

        P0 = np.random.uniform(0.0, 5.0 * param_ini[0], nwalkers)
        rp = np.random.uniform(0.0, 5000.0, nwalkers)
        a = np.random.uniform(0.0, 5.0, nwalkers)
        b = np.random.uniform(3.0, 20.0, nwalkers)
        c = np.random.uniform(0.0, 1.1, nwalkers)
        pos_tab = np.vstack((P0, rp, a, b, c)).T
        pos_tab[-1, :] = param_ini
        pos = pos_tab.tolist()

        args_lnprob = [
            theta,
            Tx_data,
            Tx_data_err,
            ne_fit,
            ne_fit_erru,
            ne_fit_errd,
            r_test,
            VPM_param,
        ]
        sampler = mcmc_sampler_pe(pos, ndim, nwalkers, nsteps, args_lnprob)

        chi2_chains = -2.0 * sampler.lnprobability
        min_chi2_chains = np.argsort(chi2_chains[:, -1])
        param_chains = sampler.chain

        ndim, nwalkers, nsteps = param_ini.size, 15, 10000
        pos = param_chains[min_chi2_chains[0:nwalkers], -1, :].tolist()
        sampler = mcmc_sampler_pe(pos, ndim, nwalkers, nsteps, args_lnprob)

        np.savez(file_chains, param=sampler.chain, lnprob=sampler.lnprobability)


def clean_chains(res_dir, ext):
    """
    Cleans the MCMC chains: removes burn-in, performs
    sigma-clipping based on posterior values, and keep
    samples seperated by auto-correlation length.
    Also saves the best-fit parameters

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    ext: string extension of the MCMC directory (either *ne* for the
    density model or *pe* for the pressure profile fit)

    Returns
    _______
    Create a .npz file in the *MCMC_ext* folder of the *results*
    directory in res_dir containing the cleaned chains.
    Also creates a .npy file in the same folder with the
    best-fit parameters

    """

    mer_dir = res_dir + "/results/"
    mcmc_dir = mer_dir + "MCMC_" + ext + "/"

    print(colored("Cleaning the chains...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_cleaned_chains = mcmc_dir + "MCMC_chains_clean.npz"
    file_best_fit = mcmc_dir + "best_fit_params.npy"

    if os.path.exists(file_best_fit):
        print(colored("MCMC chains already cleaned", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        file_chains = mcmc_dir + "MCMC_chains.npz"
        chains = np.load(file_chains)
        param_chains = chains["param"]
        chi2_chains = -2.0 * chains["lnprob"]

        nwalkers, nsteps, ndim = param_chains.shape
        burn_in = int(nsteps / 2)

        most_chi2 = np.median(
            [value for value in chi2_chains[:, nsteps - 1] if value != np.inf]
        )
        min_chi2 = np.min(np.abs(chi2_chains[:, nsteps - 1]))
        sigma_chi2 = 2.0 * (np.abs(most_chi2) - np.abs(min_chi2))
        wkeep_walkers = np.where(
            np.logical_and(
                (np.abs(chi2_chains[:, burn_in]) < np.abs(most_chi2) + sigma_chi2),
                (np.abs(chi2_chains[:, burn_in]) > np.abs(most_chi2) - sigma_chi2),
            )
        )
        N_walkers_kept = np.array(wkeep_walkers).size
        if N_walkers_kept < 2:
            print(
                colored(
                    "Not enough valid walkers. Program aborted.", "red", None, ["bold"]
                )
            )
        param_kept = param_chains[wkeep_walkers, burn_in:, :]
        param_kept = param_kept[0, :, :, :]
        chi2_kept = chi2_chains[wkeep_walkers, burn_in:]
        chi2_kept = chi2_kept[0, :, :]

        tau_tab = np.zeros((ndim, N_walkers_kept))
        for ip in range(ndim):
            for iw in range(N_walkers_kept):
                tau = emcee.autocorr.integrated_time(
                    param_kept[iw, :, ip], tol=1, quiet=True
                )
                tau_tab[ip, iw] = tau[0]

        tau_min = int(np.nanmin(tau_tab))
        param_conv = param_kept[:, ::tau_min, :]
        chi2_conv = chi2_kept[:, ::tau_min]

        samples = param_conv.reshape((-1, ndim))
        w_min_chi2 = np.where(np.abs(chi2_kept) == np.abs(chi2_kept).min())
        best_fit_param = param_kept[(w_min_chi2[0])[0], (w_min_chi2[1])[0], :]

        np.savez(file_cleaned_chains, param=param_conv, lnprob=chi2_conv, samp=samples)
        np.save(file_best_fit, best_fit_param)


def get_asymmetric_err(r, best_fit, stored_prof):
    """
    Get asymmetric error bars around the best-fit model

    Parameters
    __________
    r: the radius array associated with the best-fit model
    best_fit: the best-fit model
    stored_prof: Monte Carlo realizations of models

    Returns
    _______
    err_u, err_d: the errors bars above and below
    the best-fit model

    """

    err_u = np.zeros(r.size)
    err_d = np.zeros(r.size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(r.size):
            wpos = np.where(stored_prof[i, :] - best_fit[i] > 0)
            wneg = np.where(stored_prof[i, :] - best_fit[i] < 0)
            err_u[i] = mad(
                np.concatenate(
                    (
                        stored_prof[i, wpos[0]] - best_fit[i],
                        -1.0 * (stored_prof[i, wpos[0]] - best_fit[i]),
                    )
                )
            )
            err_d[i] = mad(
                np.concatenate(
                    (
                        stored_prof[i, wneg[0]] - best_fit[i],
                        -1.0 * (stored_prof[i, wneg[0]] - best_fit[i]),
                    )
                )
            )

    return [err_u, err_d]


def best_ne_model(res_dir, Rproj, r_map, los_step_map, z, R500):
    """
    Get best-fit density model and surface brightness
    model and estimate uncertainties

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    Rproj: a 1D array that gives the projected radius
    considered to build the 2D array los_step_map
    r_map: a 2D array used to compute the ICM
    density distribution
    los_step_map: a 2D array used to compute the
    integral of the ICM density distribution
    along the line of sight
    z: the cluster redshift
    R500: the cluster R500 radius in kpc

    Returns
    _______
    Create a .npz file in the *MCMC_ne* folder of the *results*
    directory in res_dir containing the best-fit density and surface
    brightness profiles and their associated uncertainties at 1-sigma

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    mcmc_dir = mer_dir + "MCMC_ne/"

    print(colored("Computing best-fit density profile...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_save_ne = mcmc_dir + "ne_best_fit.npz"

    if os.path.exists(file_save_ne):
        print(colored("Best-fit model already computed", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        file_best_fit = mcmc_dir + "best_fit_params.npy"
        best_fit_param = np.load(file_best_fit)

        file_cleaned_chains = mcmc_dir + "MCMC_chains_clean.npz"
        cleaned_chains = np.load(file_cleaned_chains)
        samples = cleaned_chains["samp"]

        tab_r = np.logspace(0.0, 4.0, 1000)
        ne_bfit = VPM_model(tab_r, best_fit_param[:-1])

        saved_xsb_prof_file = mer_dir + "cl_bkg_sb_profile.npz"
        saved_xsb_prof = np.load(saved_xsb_prof_file)
        theta_arcmin = saved_xsb_prof["r"]
        bkg_prof = saved_xsb_prof["xsb_bkg"]
        XSB_float_to_N = saved_xsb_prof["f2n"]

        d_a = cosmo.angular_diameter_distance(z).to("kpc").value
        theta = d_a * (theta_arcmin * u.arcmin).to("radian").value

        file_conv_tab = cl_dir + "conv_tab.npy"
        conv_fact = np.load(file_conv_tab)

        N_MC = 500
        store_profiles_ne = np.zeros((tab_r.size, N_MC))
        store_profiles_Sx = np.zeros((theta.size, N_MC))

        for i in range(N_MC):
            buff_param = samples[int(np.random.uniform(0, samples[:, 0].size)), :]
            store_profiles_ne[:, i] = VPM_model(tab_r, buff_param[:-1])
            store_profiles_Sx[:, i] = XSB_model(
                buff_param,
                theta,
                bkg_prof,
                XSB_float_to_N,
                conv_fact,
                Rproj,
                r_map,
                los_step_map,
                R500,
            )

        ne_std_up, ne_std_down = get_asymmetric_err(tab_r, ne_bfit, store_profiles_ne)

        Sx_bfit = np.median(store_profiles_Sx, axis=1)
        Sx_std_up, Sx_std_down = get_asymmetric_err(theta, Sx_bfit, store_profiles_Sx)

        np.savez(
            file_save_ne,
            r=tab_r,
            theta=theta,
            ne=ne_bfit,
            ne_erru=ne_std_up,
            ne_errd=ne_std_down,
            Sx=Sx_bfit,
            Sx_erru=Sx_std_up,
            Sx_errd=Sx_std_down,
        )


def best_icm_models(res_dir, z, R500, N_ann, Ysz):
    """
    Get best-fit ICM models and estimate uncertainties

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    z: the cluster redshift
    R500: the cluster R500 radius in kpc
    N_ann: number of annuli considered for spectrum extraction

    Returns
    _______
    Create a .npz file in the *ICM* folder of the *results*
    directory in res_dir containing the best-fit ICM profiles
    and their associated uncertainty at 1-sigma

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    mcmc_dir_ne = mer_dir + "MCMC_ne/"
    mcmc_dir_pe = mer_dir + "MCMC_pe/"

    icm_dir = mer_dir + "ICM/"

    if not os.path.exists(icm_dir):
        sp.call("mkdir " + icm_dir, shell=True)

    print(colored("Computing best-fit ICM profiles...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_save_icm = icm_dir + "ICM_best_fits.npz"

    if os.path.exists(file_save_icm):
        print(colored("Best-fit models already computed", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        bestfit_ne_file = mcmc_dir_ne + "ne_best_fit.npz"
        bestfit_ne = np.load(bestfit_ne_file)
        tab_r = bestfit_ne["r"]
        ne = bestfit_ne["ne"]
        ne_erru = bestfit_ne["ne_erru"]
        ne_errd = bestfit_ne["ne_errd"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if N_ann > 2:
                file_cleaned_chains = mcmc_dir_pe + "MCMC_chains_clean.npz"
                cleaned_chains = np.load(file_cleaned_chains)
                samples = cleaned_chains["samp"]
                file_best_fit_pe = mcmc_dir_pe + "best_fit_params.npy"
                best_fit_param_pe = np.load(file_best_fit_pe)
                pe = gNFW_model(tab_r, best_fit_param_pe)
                te = pe / ne
            else:
                Tx_fit_file = cl_dir + "T_prof_fit.npz"
                Tx_fit = np.load(Tx_fit_file)
                Tx_r = Tx_fit["fitx"]
                Tx = Tx_fit["fity"]
                Tx_err = Tx_fit["fityerr"]
                te_pow = extrap(np.log10(tab_r), np.log10(Tx_r), np.log10(Tx))
                te = 10.0 ** te_pow
                te_erru_pow = extrap(np.log10(tab_r), np.log10(Tx_r), np.log10(Tx_err))
                te_erru = 10.0 ** te_erru_pow
                te_errd_pow = extrap(np.log10(tab_r), np.log10(Tx_r), np.log10(Tx_err))
                te_errd = 10.0 ** te_errd_pow
                pe = ne * te

            ke = pe / ne ** (5.0 / 3.0)

            logT, loglambda = read_ds_cooling("m-05.cie")
            Temp_tab = (10 ** logT) * const.k_B.to(u.keV / u.K).value
            ICM_lambda = 10 ** (np.interp(te, Temp_tab, loglambda))
            te_erg = (te * u.keV).to("erg").value
            nI = ne / 1.199
            tcool = (
                1e-9
                * ((1.5 * (ne + nI) * te_erg / (ne * nI * ICM_lambda)) * u.s)
                .to("year")
                .value
            )

        delta_r = np.roll(tab_r, -1) - tab_r

        N_MC = 1500
        store_profiles_pe = np.zeros((tab_r.size, N_MC))
        store_profiles_te = np.zeros((tab_r.size, N_MC))
        store_profiles_ke = np.zeros((tab_r.size, N_MC))
        store_profiles_Mhse = np.zeros((tab_r.size, N_MC))
        store_profiles_Mgas = np.zeros((tab_r.size, N_MC))
        store_profiles_tcool = np.zeros((tab_r.size, N_MC))
        store_MYx = np.zeros(N_MC)

        kpc2m = (u.kpc).to("m")
        kev2pa = (u.keV / u.cm ** 3).to("Pa")
        sol_mass = (u.M_sun).to("kg")
        mp = const.m_p.value
        GN = const.G.value
        mu = 0.62
        mue = 1.15

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if N_ann > 2:
                P0, rp, a, b, c = best_fit_param_pe
                dpdr = (
                    -P0
                    * (tab_r / rp) ** (-c)
                    * (1 + (tab_r / rp) ** a) ** (-(a + b - c) / a)
                    * (b * (tab_r / rp) ** a + c)
                    / tab_r
                )
            else:
                dpdr = np.gradient(pe, tab_r)

            Mhse = (
                -((tab_r * kpc2m) ** 2)
                * (kev2pa * dpdr / kpc2m)
                / (mu * mp * GN * ne * 1e6)
                / sol_mass
            )
            Mgas = np.cumsum(
                ((4 * np.pi * (tab_r * kpc2m) ** 2 * ne * 1e6 * mue * mp) / sol_mass)
                * delta_r
                * kpc2m
            )

            for i in range(N_MC):
                if np.random.uniform(-1, 1) < 0:
                    ne_buff = ne - np.abs(np.random.normal()) * ne_errd
                else:
                    ne_buff = ne + np.abs(np.random.normal()) * ne_erru

                if N_ann > 2:
                    buff_param = samples[
                        int(np.random.uniform(0, samples[:, 0].size)), :
                    ]
                    P0, rp, a, b, c = buff_param
                    pe_buff = gNFW_model(tab_r, buff_param)
                    te_buff = pe_buff / ne_buff
                    dpdr = (
                        -P0
                        * (tab_r / rp) ** (-c)
                        * (1 + (tab_r / rp) ** a) ** (-(a + b - c) / a)
                        * (b * (tab_r / rp) ** a + c)
                        / tab_r
                    )
                else:
                    te_buff = te + np.random.normal() * te_erru
                    pe_buff = ne_buff * te_buff
                    dpdr = np.gradient(pe_buff, tab_r)

                ke_buff = pe_buff / ne_buff ** (5.0 / 3.0)
                Mhse_buff = (
                    -((tab_r * kpc2m) ** 2)
                    * (kev2pa * dpdr / kpc2m)
                    / (mu * mp * GN * ne_buff * 1e6)
                    / sol_mass
                )
                Mgas_buff = np.cumsum(
                    (
                        (4 * np.pi * (tab_r * kpc2m) ** 2 * ne_buff * 1e6 * mue * mp)
                        / sol_mass
                    )
                    * delta_r
                    * kpc2m
                )

                ICM_lambda_buff = 10 ** (np.interp(te_buff, Temp_tab, loglambda))
                te_erg_buff = (te_buff * u.keV).to("erg").value
                nI_buff = ne_buff / 1.199
                tcool_buff = (
                    1e-9
                    * (
                        (
                            1.5
                            * (ne_buff + nI_buff)
                            * te_erg_buff
                            / (ne_buff * nI_buff * ICM_lambda_buff)
                        )
                        * u.s
                    )
                    .to("year")
                    .value
                )

                T_mean500 = np.nanmean(
                    te_buff[((tab_r > 0.15 * R500) & (tab_r < R500))]
                )
                Mg500 = np.interp(R500, tab_r, Mgas_buff)
                Yx = Mg500 * T_mean500
                # Scaling relation from Arnaud et al. 2010 Eq. (2)
                M_Yx = (
                    10 ** 14.567
                    * (Yx / (2e14 * (cosmo.H0.value / 70.0) ** (-5.0 / 2.0))) ** 0.561
                    * (cosmo.H0.value / 70.0) ** (-1.0)
                ) / cosmo.efunc(z) ** (2.0 / 5.0)

                store_profiles_pe[:, i] = pe_buff
                store_profiles_te[:, i] = te_buff
                store_profiles_ke[:, i] = ke_buff
                store_profiles_Mhse[:, i] = Mhse_buff
                store_profiles_Mgas[:, i] = Mgas_buff
                store_profiles_tcool[:, i] = tcool_buff
                store_MYx[i] = M_Yx

        pe_std_up, pe_std_down = get_asymmetric_err(tab_r, pe, store_profiles_pe)
        te_std_up, te_std_down = get_asymmetric_err(tab_r, te, store_profiles_te)
        ke_std_up, ke_std_down = get_asymmetric_err(tab_r, ke, store_profiles_ke)
        Mhse_std_up, Mhse_std_down = get_asymmetric_err(
            tab_r, Mhse, store_profiles_Mhse
        )
        Mgas_std_up, Mgas_std_down = get_asymmetric_err(
            tab_r, Mgas, store_profiles_Mgas
        )
        tcool_std_up, tcool_std_down = get_asymmetric_err(
            tab_r, tcool, store_profiles_tcool
        )

        MYx_mean = np.nanmean(store_MYx)
        R500_Yx = (
            (
                (
                    (MYx_mean * u.Msun)
                    / ((4.0 / 3.0) * np.pi * 500.0 * cosmo.critical_density(z))
                )
                ** (1.0 / 3.0)
            )
            .to("kpc")
            .value
        )
        Delta_R500 = np.abs(R500_Yx - R500) / R500
        while Delta_R500 > 5e-2:
            New_R500 = R500_Yx
            for i in range(store_MYx.size):
                rdval = np.random.uniform(-1, 1)
                if rdval > 0:
                    Te_buff = te + np.abs(np.random.normal()) * te_std_up
                else:
                    Te_buff = te - np.abs(np.random.normal()) * te_std_down
                T_mean500 = np.nanmean(
                    Te_buff[((tab_r > 0.15 * R500) & (tab_r < R500))]
                )
                Mgas_buff = store_profiles_Mgas[:, i]
                Mg500 = np.interp(New_R500, tab_r, Mgas_buff)
                Yx = Mg500 * T_mean500
                M_Yx = (
                    10 ** 14.567
                    * (Yx / (2e14 * (cosmo.H0.value / 70.0) ** (-5.0 / 2.0))) ** 0.561
                    * (cosmo.H0.value / 70.0) ** (-1.0)
                ) / cosmo.efunc(z) ** (2.0 / 5.0)
                store_MYx[i] = M_Yx
            MYx_mean = np.nanmean(store_MYx)
            R500_Yx = (
                (
                    (
                        (MYx_mean * u.Msun)
                        / ((4.0 / 3.0) * np.pi * 500.0 * cosmo.critical_density(z))
                    )
                    ** (1.0 / 3.0)
                )
                .to("kpc")
                .value
            )
            Delta_R500 = np.abs(R500_Yx - New_R500) / New_R500

        MYx_mean = np.nanmean(store_MYx)
        MYx_std = np.nanstd(store_MYx)
        R500_Yx = (
            (
                (
                    (MYx_mean * u.Msun)
                    / ((4.0 / 3.0) * np.pi * 500.0 * cosmo.critical_density(z))
                )
                ** (1.0 / 3.0)
            )
            .to("kpc")
            .value
        )

        np.savez(
            file_save_icm,
            r=tab_r,
            ne=ne,
            ne_erru=ne_erru,
            ne_errd=ne_errd,
            pe=pe,
            pe_erru=pe_std_up,
            pe_errd=pe_std_down,
            te=te,
            te_erru=te_std_up,
            te_errd=te_std_down,
            ke=ke,
            ke_erru=ke_std_up,
            ke_errd=ke_std_down,
            Mhse=Mhse,
            Mhse_erru=Mhse_std_up,
            Mhse_errd=Mhse_std_down,
            Mgas=Mgas,
            Mgas_erru=Mgas_std_up,
            Mgas_errd=Mgas_std_down,
            tcool=tcool,
            tcool_erru=tcool_std_up,
            tcool_errd=tcool_std_down,
            R500=R500_Yx,
            MYx=MYx_mean,
            MYx_err=MYx_std,
        )

        if Ysz is not None:
            cluster_header = fits.Header()
            cluster_header["Redshift"] = z
            cluster_header["R500"] = R500_Yx
            cluster_header["Reso"] = 10.0
            cluster_header["Y075"] = Ysz[0]
            cluster_header["Y075_err"] = Ysz[1]
            cluster_header["Thetamax"] = 2.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits.writeto(
                    cl_dir + "/y_profile_MCMC.fits",
                    np.ones(10),
                    header=cluster_header,
                    overwrite=True,
                )
                fits.append(
                    cl_dir + "/y_profile_MCMC.fits",
                    np.ones(10),
                    header=cluster_header,
                    overwrite=True,
                )
                fits.append(
                    cl_dir + "/y_profile_MCMC.fits",
                    np.ones(10),
                    header=cluster_header,
                    overwrite=True,
                )


def cooling_lum(
    res_dir, z, tcool_th, Xdepro, Ydepro, multiobs, obsids, input_XSZ_file, do_err
):
    """
    Measure the X-ray luminosity within the cooling radius

    Parameters
    __________
    res_dir: the result directory named after the cluster name
    z: the cluster redshift
    tcool_th: the cooling threshold considered for the cooling radius
    Xdepro: the RA position of the deprojection center
    Ydepro: the Dec position of the deprojection center
    multiobs: are there multiple obsids to consider? True/False
    obsids: the list of obsids given as a comma-separated string of numbers
    input_XSZ_file: the fits file containing the profiles estimated from a joint X-ray/SZ analysis
    do_err: True if you want to compute the lower bound on the cooling luminosity

    Returns
    _______
    Create a .npz file in the *ICM* folder of the *results*
    directory in res_dir containing the cooling luminosity
    as well as its 1-sigma uncertainty

    """

    mer_dir = res_dir + "/results/"
    cl_dir = mer_dir + "cluster/"
    bkg_dir = mer_dir + "background/"
    icm_dir = mer_dir + "ICM/"
    fig_dir = mer_dir + "figures/"

    print(colored("Computing cooling luminosity...", "blue", None, ["bold"]))
    print("------------------------------------------------------------")

    file_save_Lcool = icm_dir + "Lcool.npz"

    if os.path.exists(file_save_Lcool):
        print(colored("Cooling luminosity already computed", "white", None, ["bold"]))
        print("------------------------------------------------------------")
    else:
        if input_XSZ_file is not None:
            hdu = fits.open(input_XSZ_file)
            rad = hdu[0].data
            te = hdu[4].data
            te_erru = hdu[5].data
            te_errd = hdu[6].data
            tc = hdu[13].data
            tc_erru = hdu[14].data
            tc_errd = hdu[15].data
        else:
            icm_file = icm_dir + "ICM_best_fits.npz"
            ICM = np.load(icm_file)
            rad = ICM["r"]
            tc = ICM["tcool"]
            tc_erru = ICM["tcool_erru"]
            tc_errd = ICM["tcool_errd"]
            te = ICM["te"]
            te_erru = ICM["te_erru"]
            te_errd = ICM["te_errd"]

        if np.nanmin(tc) < tcool_th < np.nanmax(tc):
            rcool_best = 10.0 ** np.interp(
                np.log10(tcool_th), np.log10(tc), np.log10(rad)
            )
            te_rcool_best = np.mean(te[rad <= rcool_best])

            N_MC = 1000
            tab_rcool = np.zeros(N_MC)
            tab_te = np.zeros(N_MC)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for k in range(N_MC):
                    sign = np.random.choice([-1.0, 1.0])
                    if sign < 0:
                        tc_k = tc - np.abs(np.random.normal()) * tc_errd
                        te_k = te - np.abs(np.random.normal()) * te_errd
                    else:
                        tc_k = tc + np.abs(np.random.normal()) * tc_erru
                        te_k = te + np.abs(np.random.normal()) * te_erru

                    if np.nanmin(tc_k) < tcool_th < np.nanmax(tc_k):
                        rcool = 10.0 ** np.interp(
                            np.log10(tcool_th), np.log10(tc_k), np.log10(rad)
                        )
                        te_rcool = np.mean(te_k[rad <= rcool])
                    else:
                        rcool = np.nan
                        te_rcool = np.nan

                    tab_rcool[k] = rcool
                    tab_te[k] = te_rcool

                wpos = np.where(tab_rcool - rcool_best > 0)
                wneg = np.where(tab_rcool - rcool_best < 0)
                rcool_erru = mad(
                    np.concatenate(
                        (
                            tab_rcool[wpos[0]] - rcool_best,
                            -1.0 * (tab_rcool[wpos[0]] - rcool_best),
                        )
                    )
                )
                rcool_errd = mad(
                    np.concatenate(
                        (
                            tab_rcool[wneg[0]] - rcool_best,
                            -1.0 * (tab_rcool[wneg[0]] - rcool_best),
                        )
                    )
                )

                wpos = np.where(tab_te - te_rcool_best > 0)
                wneg = np.where(tab_te - te_rcool_best < 0)
                te_rcool_erru = mad(
                    np.concatenate(
                        (
                            tab_te[wpos[0]] - te_rcool_best,
                            -1.0 * (tab_te[wpos[0]] - te_rcool_best),
                        )
                    )
                )
                te_rcool_errd = mad(
                    np.concatenate(
                        (
                            tab_te[wneg[0]] - te_rcool_best,
                            -1.0 * (tab_te[wneg[0]] - te_rcool_best),
                        )
                    )
                )

            nan_count = np.isnan(tab_rcool).sum()
            # 1-sigma confidence level <==> 34.135% of MC samples on both sides of the mean
            # You can have NaNs from infinity down to 1-sigma <==> 50% - 34.135%
            # If there are NaNs within the 1-sigma uncertainty then rcool is compatible with 0 kpc
            if nan_count > 0.15865 * N_MC:
                rcool_errd = rcool_best
                te_rcool_errd = te_rcool_best

            if do_err:
                rcool_best -= rcool_errd
                if rcool_best < 10:
                    rcool_best = 10.0

                te_rcool_best += te_rcool_erru

            # Convert cooling radius from kpc to pixel number
            map_file = mer_dir + "wide_broad_thresh_nopts.img"
            hdu = fits.open(map_file)
            cl_header = hdu[0].header
            d_a = cosmo.angular_diameter_distance(z).to("kpc").value
            rcool_pix = (
                ((rcool_best / d_a) * u.rad).to("arcsec")
                / (cl_header["CDELT2"] * 3600.0 / cl_header["CDELT2P"])
            ).value

            rcool_reg_name = cl_dir + "rcool_disk.reg"
            rcool_reg = open(rcool_reg_name, "w")
            rcool_reg.write("# Region file format: CIAO version 1.0\n")
            rcool_reg.write(
                "annulus("
                + str(Xdepro)
                + ","
                + str(Ydepro)
                + ","
                + str(0.0)
                + ","
                + str(rcool_pix)
                + ")"
            )
            rcool_reg.close()
            time.sleep(1)

            stowed_files = mer_dir + "stowed_files.txt"
            with open(stowed_files) as f:
                content = f.readlines()
            bkg_stowed_file = content[0][:-1]
            bkg_stowed_reg = content[1][:-1]

            tab_obsid = obsids.split(",")
            keep_obsid = []
            valid_spec = 0
            for obsid in tab_obsid:
                if multiobs:
                    efile = mer_dir + "All_" + obsid + "_reproj_evt_nopts.fits"
                else:
                    efile = mer_dir + "efile_repro_raw_clean_nopts.fits"

                out_file = cl_dir + "rcool_spectrum_" + obsid
                sp.call(
                    [
                        "bash",
                        "shell/extract_spectrum.sh",
                        efile,
                        rcool_reg_name,
                        bkg_stowed_file,
                        bkg_stowed_reg,
                        out_file,
                    ]
                )
                time.sleep(1)
                if os.path.exists(out_file + ".pi"):
                    valid_spec += 1
                    keep_obsid.append(obsid)

            if (valid_spec < 1) | (rcool_best < 10.0):
                print(
                    colored("No X-ray count in cooling region", "red", None, ["bold"])
                )
            else:
                efile = mer_dir + "efile_repro_raw_clean_nopts.fits"
                efile_in = efile + "[bin sky=@" + rcool_reg_name + "]"
                b_reg_name = mer_dir + "bkg_region_" + str(tab_obsid[0]) + ".reg"
                efile_b = efile + "[bin sky=@" + b_reg_name + "]"
                file_out = mer_dir + "rcool_disk_stat.fits"
                sp.call(
                    [
                        "bash",
                        "shell/extract_content_with_bkg.sh",
                        efile_in,
                        file_out,
                        efile_b,
                    ]
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hdu = fits.open(file_out)
                    tab_area_cl = hdu[1].data["AREA"]
                    c_rate = hdu[1].data["NET_RATE"]

                logger = logging.getLogger("sherpa")
                logger.setLevel(logging.WARN)

                tab_area_bkg = [0]
                SNR_bin = 3

                ui.clean()
                fit_ind = 1
                for obsid in keep_obsid:
                    file_area_bkg = mer_dir + "bkg_area_" + obsid + ".txt"
                    with open(file_area_bkg) as f:
                        content = f.readlines()
                    area_bkg = float(content[0])
                    tab_area_bkg.append(area_bkg)
                    ui.load_data(fit_ind, cl_dir + "rcool_spectrum_" + obsid + ".pi")
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

                ui.notice(0.7, 7)
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
                    ui.xsapec.kt.kt = te_rcool_best
                    ui.freeze(ui.xsapec.kt.kt)
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
                            + tab_area_fact[0]
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
                    ui.covar(kt.norm)

                i = 1
                for obsid in keep_obsid:
                    tab_area_fact = tab_area_cl / tab_area_bkg[i]
                    spec_fit = copy.deepcopy(ui.get_fit_plot(i))
                    bkg_fit = copy.deepcopy(ui.get_fit_plot(i + 1))
                    i += 2

                    int_bkg_fit = np.interp(
                        spec_fit.modelplot.x, bkg_fit.modelplot.x, bkg_fit.modelplot.y
                    )
                    file_save = fig_dir + "rcool_spec_" + obsid + ".pdf"
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
                            plt.rcParams["text.latex.preamble"] = r"\boldmath"
                            ax0.set_xlim(0.7, 7)
                            ax0.set_ylabel(
                                r"$\mathrm{Counts~s^{-1}~keV^{-1}}$", fontsize=12
                            )
                            if len(int_bkg_fit[spec_fit.modelplot.x < 7.0]) > 0:
                                ax0.set_ylim(
                                    np.max(
                                        [
                                            1e-5,
                                            0.9
                                            * np.min(
                                                float(tab_area_fact[0])
                                                * int_bkg_fit[
                                                    spec_fit.modelplot.x < 7.0
                                                ]
                                            ),
                                        ]
                                    ),
                                    1.1
                                    * np.max(
                                        spec_fit.dataplot.y + spec_fit.dataplot.yerr
                                    ),
                                )
                            else:
                                ax0.set_ylim(
                                    1e-5,
                                    1.1
                                    * np.max(
                                        spec_fit.dataplot.y + spec_fit.dataplot.yerr
                                    ),
                                )
                            ax0.grid(alpha=0.5, which="both")
                            ax0.set_xticklabels([])
                            ax0.plot(
                                spec_fit.modelplot.x,
                                spec_fit.modelplot.y,
                                color="#BA094A",
                                lw=1.7,
                                label=r"$\mathrm{Total}$",
                                zorder=10,
                            )
                            ax0.plot(
                                spec_fit.modelplot.x,
                                spec_fit.modelplot.y
                                - float(tab_area_fact[0]) * int_bkg_fit,
                                color="#ED591A",
                                lw=1.4,
                                ls=(0, (3, 5)),
                                label=r"$\mathrm{Source}$",
                                dash_capstyle="round",
                                zorder=9,
                            )
                            ax0.plot(
                                bkg_fit.modelplot.x,
                                float(tab_area_fact[0]) * bkg_fit.modelplot.y,
                                color="#8C6873",
                                lw=1.4,
                                ls=(0, (3, 5, 1, 5, 1, 5)),
                                label=r"$\mathrm{Background}$",
                                dash_capstyle="round",
                                zorder=8,
                            )
                            ax0.errorbar(
                                spec_fit.dataplot.x,
                                spec_fit.dataplot.y,
                                yerr=spec_fit.dataplot.yerr,
                                xerr=spec_fit.dataplot.xerr,
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
                                bkg_fit.modelplot.x,
                                bkg_fit.modelplot.y,
                                color="#BA094A",
                                lw=1.2,
                                label=r"$\mathrm{Background~model}$",
                                zorder=10,
                            )
                            ax1.errorbar(
                                bkg_fit.dataplot.x,
                                bkg_fit.dataplot.y,
                                yerr=bkg_fit.dataplot.yerr,
                                xerr=bkg_fit.dataplot.xerr,
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
                                    (bkg_fit.dataplot.y - bkg_fit.dataplot.yerr)[
                                        bkg_fit.dataplot.x < 7.0
                                    ]
                                ),
                                1.1
                                * np.max(bkg_fit.dataplot.y + bkg_fit.dataplot.yerr),
                            )
                            ax1.set_xlim(0.7, 7)
                            ax1.set_xlabel(r"$\mathrm{Energy~[keV]}$", fontsize=12)
                            ax1.set_ylabel(
                                r"$\mathrm{Counts~s^{-1}~keV^{-1}}$", fontsize=12
                            )
                            ax1.grid(alpha=0.5, which="both")
                            if (
                                np.max(spec_fit.modelplot.y)
                                / np.max(float(tab_area_fact[0]) * bkg_fit.modelplot.y)
                                > 10
                            ):
                                leg = ax0.legend(fontsize=10, loc=4, framealpha=1)
                            else:
                                leg = ax0.legend(fontsize=10, loc=1, framealpha=1)
                            leg.set_zorder(100)
                            ax1.legend(fontsize=10, loc=1)
                            pdf.savefig()
                            plt.close()

                final_res = ui.get_covar_results()

                arf_file = cl_dir + "rcool_spectrum_" + obsid + ".arf"
                rmf_file = cl_dir + "rcool_spectrum_" + obsid + ".rmf"
                model_str = "xsapec.kt"
                param_str = (
                    "kt.redshift="
                    + str(z)
                    + ";kt.Abundanc=0.3;kt.kt="
                    + str(te_rcool_best)
                    + ";kt.norm="
                    + str(final_res.parvals[0])
                )
                outfile = cl_dir + "cooling_lum.txt"
                sp.call(
                    [
                        "bash",
                        "shell/compute_flux.sh",
                        arf_file,
                        rmf_file,
                        model_str,
                        param_str,
                        outfile,
                    ]
                )

                with open(outfile) as f:
                    content = f.readlines()

                Fcool = float(content[-1][:-1]) * c_rate
                Lcool = (
                    Fcool
                    * (4.0 * np.pi * cosmo.luminosity_distance(z).to("cm") ** 2).value
                )

                np.savez(
                    file_save_Lcool,
                    lcool=Lcool[0],
                    rcool=rcool_best,
                    rcoolerru=rcool_erru,
                    rcoolerrd=rcool_errd,
                    te=te_rcool_best,
                    teerru=te_rcool_erru,
                    teerrd=te_rcool_errd,
                )

                print("------------------------------------------------------------")
                print(
                    "Cooling luminosity within "
                    + str(int(round(rcool_best)))
                    + " kpc: "
                    + "{:.2e}".format(Lcool[0])
                    + " erg/s"
                )
        else:
            print(
                colored(
                    "Cooling time always higher than threshold", "red", None, ["bold"]
                )
            )
