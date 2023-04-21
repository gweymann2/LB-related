import healpy as hp
import healpy
import numpy as np
import sys
import platform
from scipy.optimize import OptimizeResult
import os
import argparse
from astropy.io import fits
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import use
import xpol
use('Agg')

def define_plot_resolution(ax = None):
    fig = plt.gcf() 

    DPI = fig.get_dpi()
    fig.set_size_inches(12, 8)
    if ax == None : ax = plt.gca()
    for tickLabel in ax.get_xticklabels()+ax.get_yticklabels():
        tickLabel.set_fontsize(22)
    ax.yaxis.label.set_size(22)
    ax.xaxis.label.set_size(22)
    ax.yaxis.offsetText.set_fontsize(22)
    ax.xaxis.offsetText.set_fontsize(22)

    
def masks(map_for_dust_mask, 
          per_cent_to_keep = 85, 
          smooth_mask_deg = 2, 
          apo_mask_deg = 2, 
          verbose=True):

    # smooth it 2°
    P_dust_smoothed = healpy.sphtfunc.smoothing(map_for_dust_mask, fwhm=np.radians(smooth_mask_deg)) ; del map_for_dust_mask
    if verbose : print('P map 2° smoothed') ; hp.mollview(P_dust_smoothed, unit='mK', norm='hist') ; plt.show()

    # 15% most important binary masked 
    N = int((1-per_cent_to_keep/100)*len(P_dust_smoothed))
    mask_raw = np.array(len(P_dust_smoothed)*[1])
    for i in np.argsort(P_dust_smoothed)[-N:]:
        mask_raw[i] = 0
    if verbose : print(str(per_cent_to_keep)+'% binary mask') ; hp.mollview(mask_raw, unit='mK', norm='hist') ; plt.show()

    # Apodize the binary mask by a 2° gaussian smoothing
    #apo_mask = doit y avoir un truc mieux dans xpure ?
    apo_mask = healpy.sphtfunc.smoothing(mask_raw, fwhm=np.radians(apo_mask_deg))
    if verbose : print('2° gauss apo mask') ; hp.mollview(apo_mask, unit='mK', norm='hist') ; plt.show()

    return apo_mask


# arXiv:astro-ph/0302496v4 (13)
def covariance_matrix_harmonic(alms):
    alms = np.array(alms, copy=False, order='C') # order??
    alms = alms.view(np.float64).reshape(alms.shape+(2,)) # alm complex -> alm 2*float
    if alms.ndim > 3: # pour satisfaire le bon ordre stokes, freq, ..
        alms = alms.transpose(1, 0, 2, 3)

    #cl = []
    #for i in channels:
    #    cl.append([])
    #    for j in channels:
    #        cl[i,j] = hp.alm2cl(alms[i],alms[j])
    #cl = np.array(cl)
    
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
    #print(W.shape, alms.shape)
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

from scipy.ndimage import gaussian_filter
def smooth_weights_with_splines(ilc_filter, lbins, smooth_sigma):
    ilc_filter=ilc_filter.transpose(0,2,3,1)

    #for i in range(len(ilc_filter)):
    #    for j in range(len(ilc_filter[0])):
    #        for k in range(len(ilc_filter[0,0])):
    #            integral = np.cumsum(ilc_filter[i,j,k])[lbins]
    #            tck = interpolate.splrep(lbins, integral, s=2)
    #            yder = interpolate.splev(np.arange(0, 1536), tck, der=1) # not robust
    #            ilc_filter[i,j,k]=yder
    #ilc_filter = ilc_filter.transpose(0,3,1,2)
    #ilc_filter /= np.sum(ilc_filter, axis = -1)[...,np.newaxis]

    for i in range(len(ilc_filter)):
        for j in range(len(ilc_filter[0])):
            for k in range(len(ilc_filter[0,0])):
                ilc_filter[i,j,k]=gaussian_filter(ilc_filter[i,j,k], smooth_sigma)
    ilc_filter = ilc_filter.transpose(0,3,1,2)
    ilc_filter /= np.sum(ilc_filter, axis = -1)[...,np.newaxis]
    
    return ilc_filter
    
def hilc_alm(alms, lbins=None, fsky=None, verbose = True, smooth_sigma = None):
    
    if verbose : print('compute (cov matrix == cross spectra) between freq')
    cov = covariance_matrix_harmonic(alms)                             # shape = (stokes, freq, freq, lmax)
    if verbose : print('average it on lbins')
    cov = average_on_lbins(cov, lbins)                                 # shape = (stokes, freq, freq, lmax)
    if verbose : print('inverse it')
    cov_inv = reg_inverse(cov.swapaxes(-1, -3))                        # shape = (stokes, lmax, freq, freq)
    if verbose : print('compute ilc weights as arXiv:astro-ph/0302496v4 (13)')
    e = np.ones((len(cov[0,:,0,0]),1)) # normalization condition
    ilc_filter = np.linalg.inv(e.T @ cov_inv @ e) @ e.T @ cov_inv      # shape : (stokes, lmax, 1, freq)
    if smooth_sigma!=None: ilc_filter = smooth_weights_with_splines(ilc_filter, lbins, smooth_sigma)
    #ilc_filter = e.T @ cov_inv / e.T @ cov_inv @ e      # shape : (stokes, lmax, 1, freq)
    sigma_ilc = 1 / (e.T @ cov_inv @ e)
    sigma_w = np.einsum('aiib->aib', cov)
    del cov, cov_inv

    res = OptimizeResult()
    if verbose : print('apply the weights to obtain cleaned alms as arXiv:astro-ph/0302496v4 (10)')
    res.alms = apply_harmonic_weights(ilc_filter, alms)
    res.W = ilc_filter
    res.sigma = sigma_ilc
    res.sigma_W = sigma_w

    return res


def Cl2Dl(Cl, ls): return 1e12*ls*(ls+1)/(2*np.pi)*[Cl[l] for l in ls]

def build_dic_fwhm(path_mbs):
    print('\nBuild list of channels & fwhm')
    readme = np.load(path_mbs+'/instrument_LB_IMOv1.npy', allow_pickle=True)
    dic = readme.item()
    a, b = [], []
    for key in dic.keys():
        a_el = key
        if len(a_el) == 9: a_el += '_'
        a.append(a_el)
        b.append(dic[key]['beam'])
    channel_and_fwhm = dict(zip(a, b))
    return channel_and_fwhm

def load_cmb_no_corr(path, simu, mask = None):
    filename = 'cmb_'+simu+'_PTEP_20200915_compsep.fits'
    f_cmb = os.path.join(path+'/cmb/'+simu+'/', filename)
    print('cmb')
    alms_cmb = hp.map2alm([hp.fitsfunc.read_map(f_cmb, 0)*mask, hp.fitsfunc.read_map(f_cmb, 1)*mask, hp.fitsfunc.read_map(f_cmb, 2)*mask])
    return alms_cmb


def load_coadd_noise_fg(path, simu, channel_and_fwhm, mask = None):
    #print('a')
    alms_coadd = []
    alms_noise = []
    alms_fg = []

    for a_el in channel_and_fwhm.keys():
        print(a_el, end=': ')
        fwhm = channel_and_fwhm[a_el]
        bl = hp.gauss_beam(np.radians(fwhm/60.0), 1536, pol=True)
        
        directory = path+'/noise/'+simu
        filename = [x for x in os.listdir(directory) if (a_el in x) and ('FULL' in x)][0]
        f = directory+'/'+filename
        #print('b')

        I_noise = hp.fitsfunc.read_map(f, 0)
        Q_noise = hp.fitsfunc.read_map(f, 1)
        U_noise = hp.fitsfunc.read_map(f, 2)
        #print('c')

        _alms_corr = hp.map2alm([I_noise*mask, Q_noise*mask, U_noise*mask])
        for i_alm, i_bl in zip(_alms_corr, bl.T):
            hp.almxfl(i_alm, 1.0/i_bl, inplace=True)
        alms_noise.append(_alms_corr) ; del _alms_corr
        print('noise', end=', ')
        #print('d')

        directory = path+'/coadd_signal_maps/'+simu
        filename = [x for x in os.listdir(directory) if (a_el in x)][0]
        f = directory+'/'+filename

        I_cosig = hp.fitsfunc.read_map(f, 0)
        Q_cosig = hp.fitsfunc.read_map(f, 1)
        U_cosig = hp.fitsfunc.read_map(f, 2)
        
        I_coadd = I_noise + I_cosig ; del I_noise
        Q_coadd = Q_noise + Q_cosig ; del Q_noise
        U_coadd = U_noise + U_cosig ; del U_noise

        _alms_corr = hp.map2alm([I_coadd*mask, Q_coadd*mask, U_coadd*mask]) ; del I_coadd, Q_coadd, U_coadd
        for i_alm, i_bl in zip(_alms_corr, bl.T):
            hp.almxfl(i_alm, 1.0/i_bl, inplace=True)
        alms_coadd.append(_alms_corr) ; del _alms_corr
        print('coadd', end=', ')

        directory = path+'/cmb/'+simu
        filename = [x for x in os.listdir(directory) if (a_el in x)][0]
        f = directory+'/'+filename

        I_cmb = hp.fitsfunc.read_map(f, 0)
        Q_cmb = hp.fitsfunc.read_map(f, 1)
        U_cmb = hp.fitsfunc.read_map(f, 2)

        I_foreg = I_cosig - I_cmb ; del I_cmb, I_cosig
        Q_foreg = Q_cosig - Q_cmb ; del Q_cmb, Q_cosig
        U_foreg = U_cosig - U_cmb ; del U_cmb, U_cosig
        
        _alms_corr = hp.map2alm([I_foreg*mask, Q_foreg*mask, U_foreg*mask]) ; del I_foreg, Q_foreg, U_foreg
        for i_alm, i_bl in zip(_alms_corr, bl.T):
            hp.almxfl(i_alm, 1.0/i_bl, inplace=True)
        alms_fg.append(_alms_corr) ; del _alms_corr
        #print('fg', end=', ')
        print('fg')

    alms_coadd = np.array(alms_coadd)
    alms_fg = np.array(alms_fg)
    alms_noise = np.array(alms_noise)
    
    return alms_coadd, alms_fg, alms_noise


def build_lbin(alm_per_bin):
    lbin = [0]
    lmax = 0
    while lmax < 1536:
        while lmax*(lmax+2)-lbin[-1]*(lbin[-1]-2) <= alm_per_bin:
            lmax += 1
        lmax+=1
        lbin.append(lmax)
    return np.array(lbin[:-1])

def do_hilc_on_the_sims(path_mbs, which_sims, lbins, path_output, channel_and_fwhm, mask, smooth_sigma):
    for simu in which_sims:
        simu = (4-len(str(simu)))*'0' + str(simu)
        hilc_and_apply_weights(path_mbs, simu, lbins, path_output, channel_and_fwhm, mask, smooth_sigma)
        print('Simu '+simu+' computed.\n')

def hilc_and_apply_weights(path_mbs, simu, lbins, path_output, channel_and_fwhm, mask, smooth_sigma):

    print("\n"+simu+":")
    
    alms_coadd, alms_fg, alms_noise = load_coadd_noise_fg(path_mbs, simu, channel_and_fwhm, mask)
    print('\nHILC', end=': \n')
    res = hilc_alm(alms_coadd, lbins = lbins, smooth_sigma = smooth_sigma) ; del alms_coadd
    W = res.W

    if not os.path.exists(path_output+'/'+simu): os.system('mkdir '+path_output+'/'+simu)
    if not os.path.exists(path_output+'/'+simu+'/alms'): os.system('mkdir '+path_output+'/'+simu+'/alms')

    np.save(path_output+'/'+simu+'/W', W)
    np.save(path_output+'/'+simu+'/alms/weighted_coadd_noise', res.alms) ; del res

    alms_res_noise = apply_harmonic_weights(W, alms_noise) ; del alms_noise
    np.save(path_output+'/'+simu+'/alms/weighted_noise', alms_res_noise) ; del alms_res_noise

    alms_res_fg = apply_harmonic_weights(W, alms_fg) ; del alms_fg
    np.save(path_output+'/'+simu+'/alms/weighted_fg', alms_res_fg) ; del alms_res_fg

    alms_cmb = load_cmb_no_corr(path_mbs, simu, mask)
    np.save(path_output+'/'+simu+'/alms/cmb_no_corr', alms_cmb) ; del alms_cmb
    
    return 'ok'

    
def load_and_do_mean_and_std_weights(path, freq_list, path_output, which_sims):

    all_W = []
    for i in which_sims:
        sim = (4-len(str(i)))*'0'+str(i)
        all_W.append(np.load(path+'/'+sim+'/'+'W.npy'))
        print(end = '.')
    all_W = np.array(all_W)
    all_W = all_W.transpose(0,1,4,2,3)[...,0]
    print(np.shape(all_W))

    mean, std = np.mean(all_W, axis=0), np.std(all_W, axis=0)

    np.save(path_output+'Wl/all_W', all_W, ) ; del all_W
    np.save(path_output+'Wl/mean_weights', mean, )
    np.save(path_output+'Wl/std_weights', std, )


    # a mettre dans plot all
    ls = np.arange(0,1536,1)
    plt.figure()
    i = 0
    for i, freq in enumerate(freq_list):
        plt.plot(ls, mean[1,i])
        plt.fill_between(ls, mean[1,i]-std[1,i], mean[1,i]+std[1,i], label=freq[3]+freq[-3:], alpha = 0.3)
    plt.grid()
    plt.xlim(2,1536)
    plt.semilogx()
    plt.legend(fontsize = 16, ncol=4)
    define_plot_resolution()
    plt.savefig(path_output+'/Figures/weights.pdf')
    
def load_and_do_all(path, a, path_output, which_sims, xp, cmb_only=False):

    alms = []
    for i in which_sims:
        print(end = '.')
        sim = (4-len(str(i)))*'0'+str(i)
        if cmb_only:
            alms.append(np.load(path+'/'+sim+'/alms/'+a))
        else:
            alms.append(np.load(path+'/'+sim+'/alms/'+a)[0])

    alms = np.array(alms)    
    alms_mean = np.mean(alms, axis = 0)
    alms_std = np.std(alms, axis = 0)

    fwhm_to_smooth = 70.5

    np.save(path_output+'alms/all_alms'+a, alms) 
    print(np.shape(alms))
    if xp!=None: all_Cl = [xp.get_spectra(hp.alm2map(alms[i], 512), pixwin = False)[1] for i in which_sims]
    else: all_Cl = [hp.alm2cl(alms[i]) for i in which_sims]
    np.save(path_output+'Cl/all_Cl'+a, all_Cl)
    #fwhm=np.radians(fwhm_to_smooth/60.0)
    all_maps = [hp.alm2map(alms[i], 512) for i in which_sims]
    np.save(path_output+'maps/all_maps'+a, all_maps) #; del alms


    np.save(path_output+'alms/mean_alms'+a, alms_mean, )
    if xp!=None: 
        mean_Cl = xp.get_spectra(hp.alm2map(alms_mean, 512), pixwin = False)[1]
    else:   
        mean_Cl = hp.alm2cl(alms_mean)
    np.save(path_output+'Cl/mean_Cl'+a, mean_Cl)
    #np.save(path_output+'Cl/mean_Cl'+a, xp.get_spectra(hp.alm2map(alms_mean, 512))[1], )
    #print(np.shape(alms_mean), alms_mean.dtype)
    #fwhm=np.radians(fwhm_to_smooth/60.0)
    maps_mean = hp.alm2map(alms_mean, 512) ; del alms_mean
    np.save(path_output+'maps/mean_maps'+a, maps_mean)
    hdu = fits.PrimaryHDU(maps_mean) ; del maps_mean
    if os.path.exists(path_output+'maps/mean_maps_fits_'+a[:-4]+'.fits'): os.remove(path_output+'maps/mean_maps_fits_'+a[:-4]+'.fits')
    hdu.writeto(path_output+'maps/mean_maps_fits_'+a[:-4]+'.fits') 

    
    #np.save(path_output+'alms/std_alms'+a, alms_std, )
    #print(np.shape(alms_std), alms_std.dtype)
    #np.save(path_output+'Cl/std_Cl'+a, hp.alm2cl(alms_std), )
    #maps_std = hp.alm2map(alms_std, 512, fwhm=np.radians(fwhm_to_smooth/60.0)) ; del alms_std
    #np.save(path_output+'maps/std_maps'+a, maps_std)
    #hdu = fits.PrimaryHDU(maps_std) ; del maps_std
    #hdu.writeto(path_output+'maps/std_maps_fits_'+a[:-4]+'.fits')

    
    np.save(path_output+'Cl/v2_mean_Cl'+a, np.mean(all_Cl, axis = 0))
    #fwhm=np.radians(fwhm_to_smooth/60.0)
    maps_mean = np.mean(all_maps, axis = 0) # cpu consuming mais c'est ca ou memory consuming
    np.save(path_output+'maps/v2_mean_maps'+a, maps_mean)
    #np.save(path_output+'Cl/v3_mean_Cl'+a, hp.anafast(maps_mean))
    hdu = fits.PrimaryHDU(maps_mean) ; del maps_mean
    if os.path.exists(path_output+'maps/v2_mean_maps_fits_'+a[:-4]+'.fits'): os.remove(path_output+'maps/v2_mean_maps_fits_'+a[:-4]+'.fits')
    hdu.writeto(path_output+'maps/v2_mean_maps_fits_'+a[:-4]+'.fits')

    
    np.save(path_output+'Cl/v2_std_Cl'+a, np.std(all_Cl, axis = 0))
    #fwhm=np.radians(fwhm_to_smooth/60.0)
    maps_std = np.std(all_maps, axis = 0) # cpu consuming mais c'est ca ou memory consuming
    np.save(path_output+'maps/v2_std_maps'+a, maps_std)
    #np.save(path_output+'Cl/v3_std_Cl'+a, hp.anafast(maps_std))
    hdu = fits.PrimaryHDU(maps_std) ; del maps_std
    if os.path.exists(path_output+'maps/v2_std_maps_fits_'+a[:-4]+'.fits'): os.remove(path_output+'maps/v2_std_maps_fits_'+a[:-4]+'.fits')
    hdu.writeto(path_output+'maps/v2_std_maps_fits_'+a[:-4]+'.fits')


def write_files(path_output, freq_list, which_sims, mask, path_output_to_plot, is_xpol):
    if is_xpol:
        binning = xpol.Bins.fromdeltal(2, 1535, 1)
        xp = xpol.Xpol(mask, bins=binning)
    else: xp=None
    
    #x = 1 # x = 1: E modes - x = 0: T modes - x = 2: B modes
    if not os.path.exists(path_output_to_plot): os.system('mkdir '+path_output_to_plot)
    if not os.path.exists(path_output_to_plot+'alms'): os.system('mkdir '+path_output_to_plot+'alms')
    if not os.path.exists(path_output_to_plot+'maps'): os.system('mkdir '+path_output_to_plot+'maps')
    if not os.path.exists(path_output_to_plot+'Cl'): os.system('mkdir '+path_output_to_plot+'Cl')
    if not os.path.exists(path_output_to_plot+'Wl'): os.system('mkdir '+path_output_to_plot+'Wl')
    if not os.path.exists(path_output_to_plot+'Figures'): os.system('mkdir '+path_output_to_plot+'Figures')

    np.save(path_output_to_plot+'mask', mask)
    
    #load_and_do_mean_and_std_weights(path_output, freq_list, path_output_to_plot, which_sims)

    load_and_do_all(path_output, 'weighted_coadd_noise.npy', path_output_to_plot, which_sims, xp)
    load_and_do_all(path_output, 'weighted_noise.npy', path_output_to_plot, which_sims, xp)
    load_and_do_all(path_output, 'weighted_fg.npy', path_output_to_plot, which_sims, xp)
    load_and_do_all(path_output, 'cmb_no_corr.npy', path_output_to_plot, which_sims, xp, cmb_only=True)

    print('All rdy to plot.\n')


def plot_into_figures(path_output_to_plot, xpol=True, fsky=None):

    lmin = 2
    lmax = 1535

    ls = np.arange(2,lmax+1,1)
    mean_cmb_no_corr = np.load(path_output_to_plot+'Cl/v2_mean_Clcmb_no_corr.npy')[:,-lmax+lmin-1:]
    mean_noise = np.load(path_output_to_plot+'Cl/v2_mean_Clweighted_noise.npy')[:,-lmax+lmin-1:]
    mean_fg = np.load(path_output_to_plot+'Cl/v2_mean_Clweighted_fg.npy')[:,-lmax+lmin-1:]
    v2_std_cmb_noise = np.load(path_output_to_plot+'Cl/v2_std_Clweighted_noise.npy')[:,-lmax+lmin-1:]
    v2_std_cmb_fg = np.load(path_output_to_plot+'Cl/v2_std_Clweighted_fg.npy')[:,-lmax+lmin-1:]
    filename = '/sps/litebird/Users/gweymann/mbs/Cls_Planck2018_for_PTEP_2020_r0.fits'
    hdulist = fits.open(filename)
    hdu = hdulist[1]
    lHDU = ['TEMPERATURE', 'GRADIENT', 'CURL    ', 'G-T     ']
    mean_all = np.load(path_output_to_plot+'Cl/v2_mean_Clweighted_coadd_noise.npy')[:,-lmax+lmin-1:]
    std_all = np.load(path_output_to_plot+'Cl/v2_std_Clweighted_coadd_noise.npy')[:,-lmax+lmin-1:]


    
    for i, mode in enumerate(['TT', 'EE', 'BB', 'TE', 'EB', 'TB']):
        plt.figure(figsize=(16,12))
        plt.plot(ls, mean_cmb_no_corr[i], linestyle = '-', lw=2, color='k')
        plt.plot(ls, mean_fg[i],linestyle = '-', lw=2, color='b')
        plt.plot(ls, mean_noise[i],linestyle = '-', lw=2, color='orange')
        plt.fill_between(ls, mean_cmb_no_corr[i]-np.sqrt(2/(2*ls+1))*mean_cmb_no_corr[i], mean_cmb_no_corr[i]+np.sqrt(2/(2*ls+1))*mean_cmb_no_corr[i], alpha = 0.3, color='k', label = r'$CMB_\ell^{in}$')
        plt.fill_between(ls, mean_fg[i]-v2_std_cmb_fg[i], mean_fg[i]+v2_std_cmb_fg[i], alpha = 0.3, color='b', label = r'$fg_\ell^{weighted}$')
        plt.fill_between(ls, mean_noise[i]-v2_std_cmb_noise[i], mean_noise[i]+v2_std_cmb_noise[i], color='orange',alpha = 0.3, label = r'$N_\ell^{weighted}$')
        plt.legend(fontsize=16)
        plt.loglog()
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi\ [\mu K^2]$')
        plt.xlim(lmin, lmax)
        plt.ylim(1e-7, 1e-1)
        if mode == 'BB': plt.ylim(1e-7, 1e-3)
        define_plot_resolution()
        plt.grid()
        plt.savefig(path_output_to_plot+'Figures/Cl'+mode+'.pdf')
    del mean_cmb_no_corr, mean_noise, mean_fg, v2_std_cmb_noise, v2_std_cmb_fg

    fac = 1 
    if is_xpol == False: fac = fsky

    # à rebosser
    mean_cmb_no_corr_fullsky = np.load('/sps/litebird/Users/gweymann/ilc/hilc/output_1000_100_30/to_plot/Cl/all_Clcmb_no_corr.npy')[:,:,-lmax+lmin-1:]
    all_all = np.load(path_output_to_plot+'Cl/all_Clweighted_coadd_noise.npy')[:,:,-lmax+lmin-1:]
    mean_res = 10*np.mean((all_all-mean_cmb_no_corr_fullsky)/(np.sqrt(2/(2*ls+1))*mean_cmb_no_corr_fullsky), axis=0)

    
    for i, mode in enumerate(['TT', 'EE', 'BB', 'TE', 'EB', 'TB']):
        if i<4: cl_th = hdu.data[lHDU[i]][[int(l) for l in ls]]
        else: cl_th = np.array(len(ls)*[0])
        fig, axes = plt.subplots(2,1, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        axes[0].plot(ls, cl_th, linestyle = '-', lw=2, color='orange')
        axes[0].plot(ls, mean_all[i]/fac, linestyle = '-', lw=2, color='b')
        axes[0].fill_between(ls, cl_th-np.sqrt(2/(2*ls+1))*cl_th, cl_th+np.sqrt(2/(2*ls+1))*cl_th, alpha = 0.3, color='orange', label = r'$C_\ell^{in}$')
        axes[0].fill_between(ls, (mean_all[i]-std_all[i])/fac, (mean_all[i]+std_all[i])/fac, alpha = 0.3, color='b', label = r'$C_\ell^{out}$')
        axes[0].legend(fontsize=16)
        axes[0].xaxis.set_ticks_position('bottom')
        axes[0].set_xticklabels('')
        if i<3 : 
            axes[0].loglog()
            plt.ylim(1e-5, 1e-1)
        else: 
            plt.semilogx()
        if mode == 'BB': plt.ylim(1e-7, 1e-3)
        axes[0].set_ylabel(r'$C_\ell\ [\mu K^2]$')
        axes[0].set_xlim(lmin, lmax)
        axes[0].grid()
        define_plot_resolution(axes[0])


        #axes[1].plot(ls, cl_th-cl_th,linestyle = '-', lw=2, color='orange')
        #axes[1].plot(ls, (mean_all[i]-cl_th)/cl_th,linestyle = '-', lw=2, color='b')
        #axes[1].fill_between(ls, -np.sqrt(2/(2*ls+1)), np.sqrt(2/(2*ls+1)), alpha = 0.3, color='orange', label = r'$R_\ell^{in}$')
        #axes[1].fill_between(ls, (mean_all[i]-std_all[i]-cl_th)/cl_th, (mean_all[i]+std_all[i]-cl_th)/cl_th, alpha = 0.3, color='b', label = r'$R_\ell^{out}$')
        #axes[1].semilogx()
        #axes[1].set_xlabel(r'$\ell$')
        #axes[1].set_ylabel(r'$R_\ell$')
        #axes[1].set_xlim(lmin, lmax)
        #axes[1].set_ylim(-1,1)
        #axes[1].grid()

        #axes[1].plot(ls, (mean_all[i]-cl_th)/(np.sqrt(2/(2*ls+1))*cl_th),linestyle = '-', lw=1, color='k')
        #axes[1].semilogx()
        #axes[1].set_xlabel(r'$\ell$')
        #axes[1].set_ylabel(r'$R_\ell$')
        #axes[1].set_xlim(lmin, lmax)
        #axes[1].set_ylim(-1,1)
        #axes[1].grid()

        axes[1].plot(ls, mean_res[i],linestyle = '-', lw=1, color='k', label = f'$\chi^2/dof(\ell\in[2, 500])$ = {round(np.sum(mean_res[i][2:500]**2)/(500-2), 2)}')
        axes[1].semilogx()
        axes[1].set_xlabel(r'$\ell$')
        axes[1].set_ylabel(r'$R_\ell$')
        axes[1].set_xlim(lmin, lmax)
        axes[1].set_ylim(-6,6)
        axes[1].legend(fontsize=16)
        axes[1].grid()
        
        define_plot_resolution()
        plt.savefig(path_output_to_plot+'Figures/Cl'+mode+'_2.pdf')

    
    maps_mean_fg = np.load(path_output_to_plot+'maps/v2_mean_mapsweighted_fg.npy')
    fig, (ax1, ax2) = plt.subplots(figsize=(20,12), ncols=2)
    plt.axes(ax1)
    hp.mollview(maps_mean_fg[1], unit='muK', title='Q', hold=True)
    plt.axes(ax2)
    hp.mollview(maps_mean_fg[2], unit='muK', title='U', hold=True)
    plt.tight_layout()
    plt.savefig(path_output_to_plot+'Figures/map_mean_fg.pdf')

    maps_mean_fg = np.load(path_output_to_plot+'maps/v2_std_mapsweighted_fg.npy')
    fig, (ax1, ax2) = plt.subplots(figsize=(20,12), ncols=2)
    plt.axes(ax1)
    hp.mollview(maps_mean_fg[1], unit='muK', title='Q', hold=True)
    plt.axes(ax2)
    hp.mollview(maps_mean_fg[2], unit='muK', title='U', hold=True)
    plt.tight_layout()
    plt.savefig(path_output_to_plot+'Figures/map_std_fg.pdf')

    maps_mean_fg = np.load(path_output_to_plot+'maps/v2_mean_mapsweighted_noise.npy')
    fig, (ax1, ax2) = plt.subplots(figsize=(20,12), ncols=2)
    plt.axes(ax1)
    hp.mollview(maps_mean_fg[1], unit='muK', title='Q', hold=True)
    plt.axes(ax2)
    hp.mollview(maps_mean_fg[2], unit='muK', title='U', hold=True)
    plt.tight_layout()
    plt.savefig(path_output_to_plot+'Figures/map_mean_noise.pdf')

    maps_mean_fg = np.load(path_output_to_plot+'maps/v2_std_mapsweighted_noise.npy')
    fig, (ax1, ax2) = plt.subplots(figsize=(20,12), ncols=2)
    plt.axes(ax1)
    hp.mollview(maps_mean_fg[1], unit='muK', title='Q', hold=True)
    plt.axes(ax2)
    hp.mollview(maps_mean_fg[2], unit='muK', title='U', hold=True)
    plt.tight_layout()
    plt.savefig(path_output_to_plot+'Figures/map_std_noise.pdf')    


if __name__ == "__main__":  

    print('\nall packages are successfully loaded')
    print('python version: ' + sys.version)
    print('Current OS: ' + platform.platform()+'\n')
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-a", "--alms_per_bin", help= "number of alms per bin", type = int)
    parser.add_argument("-f", "--fsky", help= "float in ]0, 1]", type = float)
    parser.add_argument("-s", "--smooth_sigma", help= "float", type = float)
    parser.add_argument("-x", "--xpoll", help= "0: no xpol, 1: xpol", type = int)

    args = parser.parse_args()
    alm_per_bin = args.alms_per_bin
    if alm_per_bin == None: alm_per_bin = 7000
    print('job launched with #alms/bin = '+ str(alm_per_bin), end=', ')
    lbins = build_lbin(alm_per_bin)

    path_mbs = '../mbs'
    fsky = args.fsky
    if fsky == None: fsky = 1
    print('and fsky = '+ str(fsky))
    if fsky == 1: 
        mask = np.ones((3145728,))
    elif fsky >= 0 and fsky<1:
        filename = 'LB_HFT_402_dust_PTEP_20200915_compsep.fits'
        f_cmb = os.path.join(path_mbs+'/foregrounds/dust/', filename)
        print('build mask')
        mapdust_P_hf = hp.fitsfunc.read_map(f_cmb, 1)**2+ hp.fitsfunc.read_map(f_cmb, 2)**2
        mask = masks(mapdust_P_hf, per_cent_to_keep = fsky*100, verbose = False)
    else: print('fsky has to be a float in ]0, 1]')


    smooth_sigma = args.smooth_sigma

    channel_and_fwhm = build_dic_fwhm(path_mbs)
    freq_list = list(channel_and_fwhm.keys())#['LFT 40 ', 'LFT 50 ', 'LFT 60 ', 'LFT 68a', 'LFT 68b', 'LFT 78a', 'LFT 78b', 'LFT 89a', 'LFT 89b', 'LFT 100', 'LFT 119', 'LFT 140', 'MFT 100', 'MFT 119', 'MFT 140', 'MFT 166', 'MFT 195', 'HFT 195', 'HFT 235', 'HFT 280', 'HFT 337', 'HFT 402', ]
    
    add_path = ''
    path_output = f'hilc/output_{alm_per_bin}_{int(fsky*100)}_{int(smooth_sigma)}'+add_path
    print('In '+path_output)
    if not os.path.exists(path_output): os.system('mkdir '+path_output)

    which_sims = list(range(0,100))
    
    #do_hilc_on_the_sims(path_mbs, which_sims, lbins, path_output, channel_and_fwhm, mask, smooth_sigma)


    is_xpol = args.xpoll
    print(is_xpol)
    if is_xpol == 1 or is_xpol==None: 
        is_xpol = True
        to_plot_add_name = '_xpol'
    else: 
        is_xpol = False
        to_plot_add_name = ''
    path_output_to_plot = path_output+'/to_plot'+to_plot_add_name+'/'

    #write_files(path_output, freq_list, which_sims, mask, path_output_to_plot, is_xpol=is_xpol)
    plot_into_figures(path_output_to_plot, is_xpol, fsky)
   

