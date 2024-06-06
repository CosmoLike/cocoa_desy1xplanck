import numpy as np

THETA_MIN      = 2.5            # Minimum angular scale (in arcminutes)
THETA_MAX      = 900.           # Maximum angular scale (in arcminutes)
N_ANGULAR_BINS = 26             # Number of angular bins

N_LENS_BINS    = 5              # Number of lens tomographic bins
N_SRC_BINS     = 5              # Number of source tomographic bins

N_XI_POWERSPECTRA = int(N_SRC_BINS * (N_SRC_BINS + 1) / 2)      # Number of power spectra

N_XI = int(N_SRC_BINS  * (N_SRC_BINS + 1) / 2 * N_ANGULAR_BINS)
N_W  = int(N_LENS_BINS * N_ANGULAR_BINS)

ggl_exclude = []
    
γt_fullmask_list = []

for i in range(N_LENS_BINS):
    for j in range(N_SRC_BINS):
        ggl_combination = (i,j)
        if ggl_combination not in ggl_exclude:
            γt_fullmask_list.append(np.ones(N_ANGULAR_BINS))
        else:
            γt_fullmask_list.append(np.zeros(N_ANGULAR_BINS))
            
γt_fullmask = np.hstack(γt_fullmask_list)        
xi_fullmask = np.ones(N_XI)
w_fullmask  = np.ones(N_W)

N_GGL = (N_LENS_BINS * N_SRC_BINS - len(ggl_exclude))* N_ANGULAR_BINS
N_DV  = N_W + N_GGL + N_XI * 2

def get_shear_calib_mask():
    shear_calib_mask = np.zeros((N_SRC_BINS, N_DV))
    ind = 0
    for i in range(N_SRC_BINS):    
        for j in range(i, N_SRC_BINS):
            if(i!=j):
                shear_calib_mask[i][ind * N_ANGULAR_BINS:(ind+1) * N_ANGULAR_BINS] = np.ones(N_ANGULAR_BINS)
                shear_calib_mask[j][ind * N_ANGULAR_BINS:(ind+1) * N_ANGULAR_BINS] = np.ones(N_ANGULAR_BINS)

                shear_calib_mask[i][N_XI + ind * N_ANGULAR_BINS:N_XI + (ind+1) * N_ANGULAR_BINS] = np.ones(N_ANGULAR_BINS)
                shear_calib_mask[j][N_XI + ind * N_ANGULAR_BINS:N_XI + (ind+1) * N_ANGULAR_BINS] = np.ones(N_ANGULAR_BINS)
            else:
                shear_calib_mask[i][ind * N_ANGULAR_BINS:(ind+1) * N_ANGULAR_BINS] = 2 * np.ones(N_ANGULAR_BINS)
                shear_calib_mask[i][N_XI + ind * N_ANGULAR_BINS:N_XI + (ind+1) * N_ANGULAR_BINS] = 2 * np.ones(N_ANGULAR_BINS)
            ind += 1

    ind = 0
    for i in range(N_LENS_BINS):
        for j in range(N_SRC_BINS):    
            ggl_combination = (i,j)
            if ggl_combination not in ggl_exclude:
                shear_calib_mask[j][2 * N_XI + ind * N_ANGULAR_BINS:2 * N_XI + (ind+1) * N_ANGULAR_BINS] = np.ones(N_ANGULAR_BINS)
                ind += 1    
            
    return shear_calib_mask

def get_galaxy_bias_mask():
    mask = np.zeros((N_LENS_BINS, N_DV))

    for i in range(N_LENS_BINS):    
        mask[i][2 * N_XI + N_GGL + i * N_ANGULAR_BINS:2 * N_XI + N_GGL + (i+1)*N_ANGULAR_BINS] = 2

    ind = 0
    for i in range(N_LENS_BINS):    
        for j in range(N_SRC_BINS):
            ggl_combination = (i,j)
            if ggl_combination not in ggl_exclude:
                mask[i][2 * N_XI + ind * N_ANGULAR_BINS:2 * N_XI + (ind+1) * N_ANGULAR_BINS] = 1
                ind += 1                
    return mask

shear_calib_mask = get_shear_calib_mask()    
bias_mask = get_galaxy_bias_mask()

np.save('./shear_calib_mask.npy', shear_calib_mask)
np.save('./bias_mask.npy', bias_mask)
