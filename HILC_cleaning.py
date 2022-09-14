from mask_beam_hilc_v2 import *

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
    print('mask done')
    
    ###############################
    # cmb only => no noise, no fg #
    ###############################
    
    masked_map_cmb = apo_mask*produce_map(['L1-040'], 
                                          seed_cmb=seed_cmb, 
                                          make_fg=False, 
                                          make_noise=False, 
                                          gaussian_smooth = False)['L1-040']
    alms_cmb = healpy.sphtfunc.map2alm(masked_map_cmb)                           
    
    
    ###########################
    # cmb, fg, noise on, hilc #
    ###########################
    
    Wl_list, sigma_list = [], []
    alms_all_list = []
    
    for sim in range(nsim):
        print('everything on: ' + str(sim+1) + '/' + str(nsim))    
        maps_dic = produce_map(channels, 
                               seed_cmb=seed_cmb,  
                               fg_models=fg_models,
                               make_noise=make_noise, 
                               seed_noise=seed_noise_list[sim],)
        masked_maps = np.array([apo_mask * maps_dic[x] 
                                for x in maps_dic])                               ; del maps_dic

        alms_corr = correct_alms(masked_maps, channels, verbose = verbose & (sim == 0))                  ; del masked_maps
        hilc = hilc_alm(alms_corr, lbins = lbins, verbose = verbose & (sim == 0))              ; del alms_corr
        alms_all_list.append(hilc.alms)                                                 
        Wl_list.append(hilc.W)                                                                
        sigma_list.append(hilc.sigma)                                                 ; del hilc

    print('all done')

    ###############################
    # noise only => no fg, no cmb #
    ###############################

    alms_noise_list = []
    
    for sim in range(nsim):
        print('noise only: ' + str(sim+1) + '/' + str(nsim))
        maps_noise_dic = produce_map(channels, 
                               make_cmb = False,  
                               make_fg = False,
                               make_noise=make_noise, 
                               seed_noise=seed_noise_list[sim],)
        masked_noise_maps = np.array([apo_mask * maps_noise_dic[x] 
                                      for x in maps_noise_dic])                     ; del maps_noise_dic
        alms_noise_corr = correct_alms(masked_noise_maps, channels, verbose = False)          ; del masked_noise_maps
        alms_noise_weighted = apply_harmonic_weights(Wl_list[sim], alms_noise_corr) ; del alms_noise_corr
        alms_noise_list.append(alms_noise_weighted)                                 ; del alms_noise_weighted
        
    print('noise only done')
          
    return {'seed_noise_list': seed_noise_list, 
            'apo_mask':apo_mask,
            'alms_cmb': alms_cmb, 
            'alms_all_list': np.array(alms_all_list)[0,...],
            'Wl_list': np.array(Wl_list)[:,1,:,0,:] ,
            'sigma_list': np.array(sigma_list)[:,1,:,0,0],
            'alms_noise_list': np.array(alms_noise_list)[0,...],
            }
