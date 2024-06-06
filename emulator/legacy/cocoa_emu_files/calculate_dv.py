import sys
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
    
n = int(sys.argv[2])    
# ============= LHS samples =================
from pyDOE import lhs

def get_lhs_samples(N_dim, N_lhs, lhs_minmax):
    unit_lhs_samples = lhs(N_dim, N_lhs)
    lhs_params = get_lhs_params_list(unit_lhs_samples, lhs_minmax)
    return lhs_params

# ================== Calculate data vectors ==========================

cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_data_vector_list(params_list, rank):
    train_params_list      = []
    train_data_vector_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        params_arr  = np.array(list(params_list[i].values()))
        data_vector = cocoa_model.calculate_data_vector(params_list[i])
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
    return train_params_list, train_data_vector_list

def get_data_vectors(params_list, comm, rank):
    local_params_list, local_data_vector_list = get_local_data_vector_list(params_list, rank)
    if rank!=0:
        comm.send([local_params_list, local_data_vector_list], dest=0)
        train_params       = None
        train_data_vectors = None
    else:
        data_vector_list = local_data_vector_list
        params_list      = local_params_list
        for source in range(1,size):
            new_params_list, new_data_vector_list = comm.recv(source=source)
            data_vector_list = data_vector_list + new_data_vector_list
            params_list      = params_list + new_params_list
        train_params       = np.vstack(params_list)    
        train_data_vectors = np.vstack(data_vector_list)        
    return train_params, train_data_vectors


print("Iteration: %d"%(n))

if(n==0):
    if(rank==0):
        lhs_params = get_lhs_samples(config.n_dim, config.n_lhs, config.lhs_minmax)
    else:
        lhs_params = None
    lhs_params = comm.bcast(lhs_params, root=0)
    params_list = lhs_params
else:
    next_training_samples = np.load(config.savedir + '/train_samples_%d.npy'%(n))
    params_list = get_params_list(next_training_samples, config.param_labels)
    
current_iter_samples, current_iter_data_vectors = get_data_vectors(params_list, comm, rank)    
    
train_samples      = current_iter_samples
train_data_vectors = current_iter_data_vectors

# ================== Train emulator ==========================
if(rank==0):
    # ================== Chi_sq cut ==========================
    def get_chi_sq_cut(train_data_vectors):
        chi_sq_list = []
        for dv in train_data_vectors:
            delta_dv = (dv - config.dv_obs)[config.mask]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
            chi_sq_list.append(chi_sq)
        chi_sq_arr = np.array(chi_sq_list)
        select_chi_sq = (chi_sq_arr < config.chi_sq_cut)
        return select_chi_sq
    # ===============================================
    select_chi_sq = get_chi_sq_cut(train_data_vectors)
    selected_obj = np.sum(select_chi_sq)
    total_obj    = len(select_chi_sq)
        # ===============================================
        
    train_data_vectors = train_data_vectors[select_chi_sq]
    train_samples      = train_samples[select_chi_sq]
    # ========================================================
    np.save(config.savedir + '/train_data_vectors_%d.npy'%(n), train_data_vectors)
    np.save(config.savedir + '/train_samples_%d.npy'%(n), train_samples)
    # ======================================================== 
    
MPI.Finalize
