import sys
from os.path import join as pjoin
from mpi4py import MPI
import numpy as np
import torch
from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler
import emcee

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

configfile = sys.argv[1]
config = Config(configfile)
label = config.emu_type.lower()
if label=="nn":
    label = label+f'{config.nn_model}'
if(rank==0):
    print("Initializing configuration space data vector dimension!")
    print("N_xip: %d"%(config.N_xi))
    print("N_xim: %d"%(config.N_xi))
    print("N_ggl: %d"%(config.N_ggl))
    print("N_w: %d"%(config.N_w))
    print("N_gk: %d"%(config.N_gk))
    print("N_sk: %d"%(config.N_sk))
    print("N_kk: %d"%(config.N_kk))
    
n = int(sys.argv[2])    
# ============= LHS samples =================
from pyDOE import lhs

def get_lhs_samples(N_dim, N_lhs, lhs_minmax):
    ''' Generate Latin Hypercube sample at parameter space
    Input:
    ======
        - N_dim: 
            Dimension of parameter space
        - N_lhs:
            Number of LH grid per dimension in the parameter space
        - lhs_minmax:
            The boundary of parameter space along each dimension
    Output:
    =======
        - lhs_params:
            LHS of parameter space
    '''
    unit_lhs_samples = lhs(N_dim, N_lhs)
    lhs_params = get_lhs_params_list(unit_lhs_samples, lhs_minmax)
    return lhs_params

# ================== Calculate data vectors ==========================

cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_data_vector_list(params_list, rank, return_s8=False):
    ''' Evaluate data vectors dispatched to the local process
    Input:
    ======
        - params_list: 
            full parameters to be evaluated. Parameters dispatched is a subset of the full parameters
        - rank: 
            the rank of the local process
    Outputs:
    ========
        - train_params: model parameters of the training sample
        - train_data_vectors: data vectors of the training sample
    '''
    train_params_list      = []
    train_data_vector_list = []
    train_sigma8_list      = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        if ((i-rank*N_local)%20==0):
            print(f'[{rank}/{size}] get_local_data_vector_list: iteration {i-rank*N_local}...')
        params_arr  = np.array(list(params_list[i].values()))
        # Here it calls cocoa to calculate data vectors at requested parameters
        data_vector, _s8 = cocoa_model.calculate_data_vector(params_list[i], return_s8=return_s8)
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
        if return_s8:
            train_sigma8_list.append(_s8)
    if return_s8:
        return train_params_list, train_data_vector_list, train_sigma8_list
    else:
        return train_params_list, train_data_vector_list, None

def get_data_vectors(params_list, comm, rank, return_s8=False):
    ''' Evaluate data vectors
    This function will further calls `get_local_data_vector_list` to dispatch jobs to and collect training data set from  other processes.
    Input:
    ======
        - params_list:
            Model parameters to be evaluated the model at
        - comm:
            MPI comm
        - rank:
            MPI rank
    Output:
    =======
        - train_params:
            model parameters of the training sample
        - train_data_vectors:
            data vectors of the training sample
    '''
    local_params_list, local_data_vector_list, local_sigma8_list = get_local_data_vector_list(params_list, rank, return_s8=return_s8)
    if rank!=0:
        comm.send([local_params_list, local_data_vector_list, local_sigma8_list], dest=0)
        train_params       = None
        train_data_vectors = None
        train_sigma8       = None
    else:
        data_vector_list = local_data_vector_list
        params_list      = local_params_list
        sigma8_list      = local_sigma8_list
        for source in range(1,size):
            new_params_list, new_data_vector_list, new_sigma8_list = comm.recv(source=source)
            data_vector_list = data_vector_list + new_data_vector_list
            params_list      = params_list + new_params_list
            sigma8_list      = sigma8_list + new_sigma8_list
        train_params       = np.vstack(params_list)    
        train_data_vectors = np.vstack(data_vector_list)
        train_sigma8       = np.vstack(sigma8_list)
    return train_params, train_data_vectors, train_sigma8

if(rank==0):
    print("Iteration: %d"%(n))
# ============== Retrieve training sample ======================
if(n==0):
    if(rank==0):
        # retrieve LHS parameters
        # the parameter space boundary is set by config.lhs_mimax, which is the
        # prior boundaries for flat prior and +- 4 sigma for Gaussian prior
        lhs_params = get_lhs_samples(config.n_dim, config.n_lhs, config.lhs_minmax)
    else:
        lhs_params = None
    lhs_params = comm.bcast(lhs_params, root=0)
    params_list = lhs_params
else:
    next_training_samples = np.load(pjoin(config.traindir, f'samples_{label}_{n}.npy'))
    params_list = get_params_list(next_training_samples, config.param_labels)
    
current_iter_samples, current_iter_data_vectors, current_iter_sigma8 = get_data_vectors(params_list, comm, rank, return_s8=True)
    
train_samples      = current_iter_samples
train_data_vectors = current_iter_data_vectors
train_sigma8       = current_iter_sigma8

# ============ Clean training data & save ====================
if(rank==0):
    # ================== Chi_sq cut ==========================
    def get_chi_sq_cut(train_data_vectors):
        chi_sq_list = []
        for dv in train_data_vectors:
            delta_dv = (dv - config.dv_obs)[config.mask]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
            chi_sq_list.append(chi_sq)
        chi_sq_arr = np.array(chi_sq_list)
        print(f'chi2 difference [{np.nanmin(chi_sq_arr)}, {np.nanmax(chi_sq_arr)}]')
        select_chi_sq = (chi_sq_arr < config.chi_sq_cut)
        return select_chi_sq
    # ===============================================
    select_chi_sq = get_chi_sq_cut(train_data_vectors)
    selected_obj = np.sum(select_chi_sq)
    total_obj    = len(train_data_vectors)
    print(f'[calculate_dv.py] Select {selected_obj} out of {total_obj}!')
    # ===============================================
        
    train_data_vectors = train_data_vectors[select_chi_sq]
    train_samples      = train_samples[select_chi_sq]
    train_sigma8       = train_sigma8[select_chi_sq]
    # ========================================================
    np.save(pjoin(config.traindir, f'data_vectors_{label}_{n}.npy'), train_data_vectors)
    np.save(pjoin(config.traindir, f'samples_{label}_{n}.npy'), train_samples)
    np.save(pjoin(config.traindir, f'sigma8_{label}_{n}.npy'), train_sigma8)
    # ======================================================== 
    print(f'Done data vector calculation iteration {n}!')
MPI.Finalize
