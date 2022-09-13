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

def produce_map(channels, make_cmb=True, gaussian_smooth = True, make_fg=True, make_noise=True, seed_noise=None, seed_cmb=None, nside=512, cmb_r=0.02, fg_models = ["pysm_ame_1","pysm_dust_0","pysm_synch_0", "pysm_freefree_1"]):
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

def masks(map_for_dust_mask, per_cent_to_keep = 85, smooth_mask_deg = 2, apo_mask_deg = 2, verbose=True):

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

def correct_alms(masked_maps, channels, verbose=True):
    
    if verbose : print('compute alms from masked maps (and plot the associated cls)')
    alms = np.array([healpy.sphtfunc.map2alm(m) for m in masked_maps])

    if verbose:
        # plot the beamed 
        plt.figure()
        for i in range(len(alms)):
            plt.plot(hp.alm2cl(alms[i, 1])[2:], label = channels[i])
        plt.semilogy()
        plt.legend(fontsize=20)
        define_plot_resolution()
        plt.show()


    if verbose : print('correct the beams by a bl (plot the bl and plot the corrected cls)')
    sim = lbs.Simulation(base_path="output")
    fwhm_channels = [lbs.FreqChannelInfo.from_imo(sim.imo,
                    "/releases/v1.0/satellite/"+channel[0]+"FT/"+channel+"/channel_info",).fwhm_arcmin 
                    for channel in channels]
    
    alms_corr = np.copy(alms)                                         ;  del alms
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
        plt.legend(fontsize=20)
        plt.show()
        
        # plot the corrected
        plt.figure()
        for i in range(len(alms_corr)):
            plt.plot(hp.alm2cl(alms_corr[i, 1])[2:], label = channels[i])
        plt.semilogy()
        define_plot_resolution()
        plt.legend(fontsize=20)
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
        plt.legend(fontsize = 20)
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


def hilc_alm(alms, channels, lbins=None, fsky=None, verbose = False):
    cl_in = np.array([hp.alm2cl(alm) for alm in alms])
    if verbose : print('compute (cov matrix == cross spectra) between freq')
    cov = covariance_matrix_harmonic(alms)                             # shape = (stokes, freq, freq, lmax)
    if verbose : print('average it on lbins')
    cov = average_on_lbins(cov, lbins)                                 # shape = (stokes, freq, freq, lmax)
    if verbose : print('inverse it')
    cov_inv = reg_inverse(cov.swapaxes(-1, -3), verbose=verbose)       # shape = (stokes, lmax, freq, freq)
    if verbose : print('compute ilc weights as arXiv:astro-ph/0302496v4 (13)')
    e = np.ones((len(channels),1)) # normalization condition
    ilc_filter = np.linalg.inv(e.T @ cov_inv @ e) @ e.T @ cov_inv      # shape : (stokes, lmax, 1, freq)
    del cov, cov_inv
    res = OptimizeResult()
    if verbose : print('apply the weights to obtain cleaned alms as arXiv:astro-ph/0302496v4 (10)')
    res.s = apply_harmonic_weights(ilc_filter, alms)                   # shape : (1, stokes, lmax*(lmax+1)/2)
    if verbose : print('compute the associated cls')
    cl_out = np.array([hp.alm2cl(alm) for alm in res.s])

    res.cl_in = cl_in
    res.cl_out = cl_out
    #res.cl_out_smooth = cl_out_smooth
    res.fsky = fsky
    res.W = ilc_filter

    return res


def Cl2Dl(Cl, ls): return 1e12*ls*(ls+1)/(2*np.pi)*[Cl[l] for l in ls]


def Cl_Wl(nsim, 
          channels, 
          fg_models, 
          make_noise, 
          per_cent_to_keep_, 
          apo_mask_deg_, 
          smooth_mask_deg_, 
          lbins, 
          seed_cmb,  
          do_map = False,
          verbose = True):
    
    seed_noise_list = np.random.randint(0, 10000, size = nsim)
    if verbose:print(''.center(116,'-'))
    if verbose:print("STEP 1 : BUILD THE MASK".center(116, ' '))
    ##################
    # build the mask #
    ##################
    
    map_for_dust_mask = produce_map(['H3-402'], 
                                    make_cmb=False, 
                                    make_noise = False, 
                                    gaussian_smooth = False, 
                                    fg_models=fg_models, 
                                    seed_cmb=seed_cmb)['H3-402']
    
    apo_mask = masks(map_for_dust_mask, 
                      per_cent_to_keep = per_cent_to_keep_, 
                      apo_mask_deg=apo_mask_deg_, 
                      smooth_mask_deg = smooth_mask_deg_, 
                      verbose = verbose)                                          ; del map_for_dust_mask
    if verbose:print('mask done'.center(116,' '));print(''.center(116,'-'))

    if verbose:print("STEP 2 : CMB ONLY MAP/Cl".center(116,' '))
    ###############################
    # cmb only => no noise, no fg #
    ###############################
    
    masked_map_cmb = apo_mask*produce_map(['L1-040'], 
                                          seed_cmb=seed_cmb, 
                                          make_fg=False, 
                                          make_noise=False, 
                                          gaussian_smooth = False)['L1-040']
    alms_cmb = healpy.sphtfunc.map2alm(masked_map_cmb)                           
    if not do_map: del masked_map_cmb
    Cl_cmb = Cl2Dl(hp.sphtfunc.alm2cl(alms_cmb)[1], ls)                          ; del alms_cmb
    if verbose:print('cmb_only done'.center(116,' '));print(''.center(116,'-'))
    
    if verbose:print("STEP 3 : CMB-FG-NOISE CLEAN MAP/Cl".center(116, ' '))
    ###########################
    # cmb, fg, noise on, hilc #
    ###########################
    
    Cl_out_mean = np.zeros((lmax,))
    Wl_mean = np.zeros((lmax,len(channels),))
    Cl_out_list, Wl_list = [], []
    
    for sim in range(nsim):
        if verbose:print('everything on: ' + str(sim+1) + '/' + str(nsim))    
        maps_dic = produce_map(channels, 
                               seed_cmb=seed_cmb,  
                               fg_models=fg_models,
                               make_noise=make_noise, 
                               seed_noise=seed_noise_list[sim],)
        masked_maps = np.array([apo_mask * maps_dic[x] 
                                for x in maps_dic])                               ; del maps_dic
        #print('ok')
        alms_corr = correct_alms(masked_maps, channels, verbose = verbose & (sim == 0))                  ; del masked_maps
        hilc = hilc_alm(alms_corr, channels, lbins = lbins, verbose = verbose & (sim == 0))              ; del alms_corr
        if (do_map & sim == 0):
            map1sim_clean = healpy.sphtfunc.alm2map(hilc.s[0], 512)
        Cl_out_mean += Cl2Dl(hilc.cl_out[0,1], ls)
        Wl_list.append(hilc.W)                                                    ; del hilc
    Wl_list = np.array(Wl_list)
    Cl_out_mean /= nsim
    if verbose:print(' cmb-noise-fg done '.center(116,' '));print(''.center(116,'-'))

    if verbose:print("STEP 4 : WEIGHTED NOISE MAP/Cl".center(116, ' '))
    ###############################
    # noise only => no fg, no cmb #
    ###############################

    Cl_out_noise_mean = np.zeros((lmax,))
    for sim in range(nsim):
        if verbose:print('noise only: ' + str(sim+1) + '/' + str(nsim))
        maps_noise_dic = produce_map(channels, 
                               make_cmb = False,  
                               make_fg = False,
                               make_noise=make_noise, 
                               seed_noise=seed_noise_list[sim],)
        masked_noise_maps = np.array([apo_mask * maps_noise_dic[x] 
                                      for x in maps_noise_dic])                       ; del maps_noise_dic
        alms_noise_corr = correct_alms(masked_noise_maps, channels, verbose = False)          ; del masked_noise_maps
        
        alms_noise_weighted = apply_harmonic_weights(Wl_list[sim], alms_noise_corr)

        if (do_map & sim == 0):
            map1sim_noise = healpy.sphtfunc.alm2map(alms_noise_weighted[0], 512)
            
        cl_out = np.array([hp.alm2cl(alm) for alm in alms_noise_weighted])
        Cl_out_noise_mean += Cl2Dl(cl_out[0,1], ls)                                   ; del alms_noise_corr
    Cl_out_noise_mean /= nsim
    
    if verbose:print('noise only done'.center(116,' '));print(''.center(116,'-'))
    if verbose:print("DONE".center(116,' '))

    return {'seed_noise_list': seed_noise_list, 
            'apo_mask':apo_mask,
            'Cl_cmb': Cl_cmb, 
            'map_cmb': masked_map_cmb,
            'Wl_list': np.array(Wl_list[:,1,:,0,:]), 
            'Cl_out_mean': Cl_out_mean, 
            'map1sim_clean': map1sim_clean,
            'Cl_out_noise_mean': Cl_out_noise_mean, 
            'map1sim_noise': map1sim_noise,
            }
