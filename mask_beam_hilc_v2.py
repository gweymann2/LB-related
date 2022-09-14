import healpy as hp
import healpy
import numpy as np
import matplotlib.pyplot as plt
print('all packages are successfully loaded')
import sys
print('python version: ' + sys.version)
import platform
print('Current OS: ' + platform.platform())
from scipy.optimize import OptimizeResult

def define_plot_resolution():
    fig = plt.gcf()  # get current figure0

    DPI = fig.get_dpi()
#     fig.set_size_inches(1920.0 / float(DPI), 1080.0 / float(DPI))
    fig.set_size_inches(12, 8)
    ax = plt.gca()
    for tickLabel in ax.get_xticklabels()+ax.get_yticklabels():
        tickLabel.set_fontsize(29)
    ax.yaxis.label.set_size(29)
    ax.xaxis.label.set_size(29)
    ax.yaxis.offsetText.set_fontsize(29)
    ax.xaxis.offsetText.set_fontsize(29)
    
    return

# Functions

import litebird_sim as lbs
import healpy

def produce_map(channels,
                make_cmb=True, 
                gaussian_smooth = True,
                make_fg=True, 
                make_noise=True, 
                seed_noise=None, 
                seed_cmb=None, 
                nside=512, 
                cmb_r=0.02, 
                fg_models = ["pysm_ame_1","pysm_dust_0","pysm_synch_0", "pysm_freefree_1"]):
                
    sim = lbs.Simulation(base_path="output")
    params = lbs.MbsParameters(
        nside = nside,
        seed_cmb=seed_cmb,
        seed_noise=seed_noise,
        make_cmb=make_cmb,
        make_fg=make_fg,
        make_noise=make_noise,
        cmb_r=cmb_r,
        gaussian_smooth = gaussian_smooth,
        fg_models=fg_models,
        save=False,
    )
    mbs = lbs.Mbs(
        simulation=sim,
        parameters=params,
        channel_list=[
            lbs.FreqChannelInfo.from_imo(
                sim.imo,
                "/releases/v1.0/satellite/"+channel[0]+"FT/"+channel+"/channel_info",
            ) for channel in channels
        ],
    )
    (healpix_maps, file_paths) = mbs.run_all()
    return healpix_maps

lmax = 1536 # nisde = 512
ls = np.arange(0, lmax)

# mask

def masks(map_for_dust_mask, 
          per_cent_to_keep = 85, 
          smooth_mask_deg = 2, 
          apo_mask_deg = 2, 
          verbose=True):

    # dust model
    channels_fordustmask = ["H3-402"]
    seed_cmb = np.random.randint(0,10000)

    # dust P map 
    P_dust = np.sqrt(map_for_dust_mask[1]**2+map_for_dust_mask[2]**2)
    if verbose : print('H3-402 P map') ; hp.mollview(P_dust, unit='mK', norm='hist') ; plt.show()
    
    # smooth it 2°
    P_dust_smoothed = healpy.sphtfunc.smoothing(P_dust, fwhm=np.radians(smooth_mask_deg))
    if verbose : print('P map 2° smoothed') ; hp.mollview(P_dust_smoothed, unit='mK', norm='hist') ; plt.show()

    # 15% most important binary masked 
    N = int((1-per_cent_to_keep/100)*len(P_dust))
    mask_raw = np.array(len(P_dust_smoothed)*[1])
    for i in np.argsort(P_dust_smoothed)[-N:]:
        mask_raw[i] = 0
    if verbose : print(str(per_cent_to_keep)+'% binary mask') ; hp.mollview(mask_raw, unit='mK', norm='hist') ; plt.show()

    # Apodize the binary mask by a 2° gaussian smoothing
    apo_mask = healpy.sphtfunc.smoothing(mask_raw, fwhm=np.radians(apo_mask_deg))
    if verbose : print('2° gauss apo mask') ; hp.mollview(apo_mask, unit='mK', norm='hist') ; plt.show()

    # multiply by the maps
    # if verbose : print('Masked U M2-119 map') ; hp.mollview(apo_mask*masked_maps[10,2], unit='mK', norm='hist') ; plt.show()
    
    return apo_mask


# beam corrected alm

def correct_alms(masked_maps, 
                 channels,
                 verbose=True):
    
    if verbose : print('compute alms from masked maps (and plot the associated cls)')
    alms = np.array([healpy.sphtfunc.map2alm(m) for m in masked_maps])

    if verbose:
        # plot the beamed 
        plt.figure()
        for i in range(len(alms)):
            plt.plot(hp.alm2cl(alms[i, 1])[2:], label = channels[i])
        plt.semilogy()
        plt.legend(fontsize=18)
        define_plot_resolution()
        plt.show()


    if verbose : print('correct the beams by a bl (plot the bl and plot the corrected cls)')
    sim = lbs.Simulation(base_path="output")
    fwhm_channels = [lbs.FreqChannelInfo.from_imo(sim.imo,
                    "/releases/v1.0/satellite/"+channel[0]+"FT/"+channel+"/channel_info",).fwhm_arcmin 
                    for channel in channels]
    
    alms_corr = np.copy(alms)                                                                           ;  del alms
    for fwhm, alm in zip(fwhm_channels, alms_corr):
        bl = hp.gauss_beam(np.radians(fwhm/60.0), lmax, pol=True)
        for i_alm, i_bl in zip(alm, bl.T):
            hp.almxfl(i_alm, 1.0/i_bl, inplace=True)
    

    if verbose:    
        
        # plot the bls
        plt.figure()
        for i, fwhm in enumerate(fwhm_channels):
            bl = hp.gauss_beam(np.radians(fwhm/60.0), lmax, pol=True)
            plt.plot(bl[:,0], label = channels[i]+' with fwhm='+str(fwhm_channels[i]))
        plt.semilogy()
        define_plot_resolution()
        plt.legend(fontsize=18)
        plt.show()
        
        # plot the corrected
        plt.figure()
        for i in range(len(alms_corr)):
            plt.plot(hp.alm2cl(alms_corr[i, 1])[2:], label = channels[i])
        plt.semilogy()
        define_plot_resolution()
        plt.legend(fontsize=18)
        plt.show()
    
    return alms_corr

# compute weights and clean alms

# arXiv:astro-ph/0302496v4 (13)

def covariance_matrix_harmonic(alms):
    alms = np.array(alms, copy=False, order='C') # order??
    alms = alms.view(np.float64).reshape(alms.shape+(2,)) # alm complex -> alm 2*float
    if alms.ndim > 3: # pour satisfaire le bon ordre stokes, freq, ..
        alms = alms.transpose(1, 0, 2, 3)
    lmax = hp.Alm.getlmax(alms.shape[-2])
    res = (alms[..., np.newaxis, :, :lmax+1, 0] * alms[..., :, np.newaxis, :lmax+1, 0]) # place pour Clij
    
    consumed = lmax + 1
    for i in range(1, lmax+1):
        n_m = lmax + 1 - i
        alms_m = alms[..., consumed:consumed+n_m, :] # va chercher les alms a un l donné
        res[..., i:] += 2 * np.einsum('...fli,...nli->...fnl', alms_m, alms_m) # somme des produits de alms pour un l
        consumed += n_m

    res /= 2 * np.arange(lmax + 1) + 1 # normalise chaque coeff pour obtenir des moyennes donc des Cl
    
    return res # matrice des Clij shape = (teb, freq, freq, lmax)

# inverse the above (en régularisant les pblms de singularité potentiels)
def reg_inverse(cov, verbose=False):
    # fgbuster way otherwise matrix singular
    inv_std = np.einsum('...ii->...i', cov) # diag de la cov
    inv_std = 1 / np.sqrt(inv_std)
    np.nan_to_num(inv_std, False, 0, 0, 0) # nan, -inf, +inf -> 0
    np.nan_to_num(cov, False, 0, 0, 0)


    if verbose:
        cond1 = []
        cond2 = []

        for i in range(2, 1536):
            cond1.append(np.linalg.cond(cov[1,i,...]))
            cond2.append(np.linalg.cond((cov * inv_std[..., np.newaxis] * inv_std[..., np.newaxis, :])[1,i,...]))

        plt.plot(range(2, 1536),cond1,label='Avec mat cov')
        plt.plot(range(2, 1536),cond2,label='Avec mat corr')
        plt.ylabel('cond number')
        plt.xlabel(r'$l$')
        plt.legend(fontsize = 18)
        define_plot_resolution()
        plt.loglog()
        plt.show()
        
    inv_cov = np.linalg.pinv(cov * inv_std[..., np.newaxis] * inv_std[..., np.newaxis, :]) # pinv
    return inv_cov * inv_std[..., np.newaxis] * inv_std[..., np.newaxis, :]


# apply the weights on the alms arXiv:astro-ph/0302496v4 (10)
def apply_harmonic_weights(W, alms):
    lmax = hp.Alm.getlmax(alms.shape[-1])
    res = np.full((W.shape[-2],) + alms.shape[1:], np.nan, dtype=alms.dtype) # rempli une array (1, 3, 1180416) de nan
    start = 0
    for i in range(0, lmax+1):
        n_m = lmax + 1 - i
        res[..., start:start+n_m] = np.einsum('...lcf,f...l->c...l', W[..., i:, :, :], alms[..., start:start+n_m]) # remplace autant de nan que besoin par les Wilm * alms
        start += n_m
    return res # shape : (1, stokes, lmax*(lmax+1)/2)


def average_on_lbins(cov, lbins):
    for lmin, lmax in zip(lbins[:-1], lbins[1:]):
        # Average the covariances in the bin
        lmax = min(lmax, cov.shape[-1])
        dof = 2 * np.arange(lmin, lmax) + 1
        cov[..., lmin:lmax] = ((dof / dof.sum() * cov[..., lmin:lmax]).sum(-1))[..., np.newaxis]
    return cov


def hilc_alm(alms, lbins=None, fsky=None, verbose = True):
    
    if verbose : print('compute (cov matrix == cross spectra) between freq')
    cov = covariance_matrix_harmonic(alms)                             # shape = (stokes, freq, freq, lmax)
    if verbose : print('average it on lbins')
    cov = average_on_lbins(cov, lbins)                                 # shape = (stokes, freq, freq, lmax)
    if verbose : print('inverse it')
    cov_inv = reg_inverse(cov.swapaxes(-1, -3))                        # shape = (stokes, lmax, freq, freq)
    if verbose : print('compute ilc weights as arXiv:astro-ph/0302496v4 (13)')
    e = np.ones((len(cov[0,:,0,0]),1)) # normalization condition
    ilc_filter = np.linalg.inv(e.T @ cov_inv @ e) @ e.T @ cov_inv      # shape : (stokes, lmax, 1, freq)
    sigma_ilc = 1 / (e.T @ cov_inv @ e)
    del cov, cov_inv

    res = OptimizeResult()
    if verbose : print('apply the weights to obtain cleaned alms as arXiv:astro-ph/0302496v4 (10)')
    res.alms = apply_harmonic_weights(ilc_filter, alms)
    res.W = ilc_filter
    res.sigma = sigma_ilc

    return res


def Cl2Dl(Cl, ls): return 1e12*ls*(ls+1)/(2*np.pi)*[Cl[l] for l in ls]


