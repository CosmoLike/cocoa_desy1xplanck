import sys
import os
from os.path import join as pjoin
from mpi4py import MPI
import numpy as np
from cocoa_emu import Config, CocoaModel, get_gaussian_samples
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
pid = os.getpid()

configfile = sys.argv[1]
config = Config(configfile)
if config.init_sample_type == "lhs":
    # label_train = f'{config.init_sample_type}_{config.n_lhs}'
    # label_valid = label_train
    print("We don't support LHS any more!")
    exit(1)
else:
    iss = f'{config.init_sample_type}'
    label_train = iss+f'_t{config.gtemp_t}_{config.gnsamp_t}'
    label_valid = iss+f'_t{config.gtemp_v}_{config.gnsamp_v}'

if(rank==0):
    print("Initializing configuration space data vector dimension!")
    print("N_xi_pm: %d"%(config.probe_size[0]))
    print("N_ggl: %d"%(config.probe_size[1]))
    print("N_w: %d"%(config.probe_size[2]))
    print("N_gk: %d"%(config.probe_size[3]))
    print("N_sk: %d"%(config.probe_size[4]))
    print("N_kk: %d"%(config.probe_size[5]))
    dump_dir = pjoin(config.traindir, 'dump')
    os.makedirs(dump_dir, exist_ok=True)

# ============== Retrieve training & validation sample ======================
# Note that training sample does not include fast parameters
training_samples_params_fn = pjoin(config.traindir,f'total_samples_{label_train}.npy')
validation_samples_params_fn = pjoin(config.traindir,f'total_samples_{label_valid}.npy')
if(rank==0):
    # retrieve Gaussian-approximation parameters
    # The mean of the Gaussian is specified by config.running_params_fid
    # plus shift from config.gauss_shift.
    if os.path.exists(training_samples_params_fn):
        print(f'[process 0]: Loading training sample from {training_samples_params_fn}...')
        params_train = np.load(training_samples_params_fn)
    else:
        print(f'[process 0]: Retrieving training sample...')
        params_train = get_gaussian_samples(config.running_params_fid, 
            config.running_params, config.params, config.gnsamp_t, 
            config.gauss_cov, config.gtemp_t, config.gshift_t)
        print(f'Saving training sample to ', training_samples_params_fn)
        np.save(training_samples_params_fn, params_train)
    if os.path.exists(validation_samples_params_fn):
        print(f'[process 0]: Loading validation sample from {validation_samples_params_fn}...')
        params_valid =  np.load(validation_samples_params_fn)
    else:
        print(f'[process 0]: Retrieving validation sample...')
        params_valid = get_gaussian_samples(config.running_params_fid, 
            config.running_params, config.params, config.gnsamp_v, 
            config.gauss_cov, config.gtemp_v, config.gshift_v)
        print(f'Saving validation sample to ', validation_samples_params_fn)
        np.save(validation_samples_params_fn, params_valid)
else:
    params_train, params_valid = None, None
params_train = comm.bcast(params_train, root=0)
params_valid = comm.bcast(params_valid, root=0)

# ================== Calculate data vectors ==========================

cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_data_vector_list(params_list, rank, label, return_s8=False):
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
    # print results real time 
    dump_file = pjoin(config.traindir, f'dump/{label}_{rank}-{size}.txt')
    fp = open(dump_file, "w")
    train_params_list      = []
    train_data_vector_list = []
    train_sigma8_list      = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        if ((i-rank*N_local)%20==0):
            print(f'[{rank}/{size}] get_local_data_vector_list: iteration {i-rank*N_local}/{N_local}...')
        if type(params_list[i]) != dict:
            _p = {k:v for k,v in zip(config.running_params, params_list[i])}
        else:
            _p = params_list[i]
        params_arr  = np.array([_p[k] for k in config.running_params])
        # Here it calls cocoa to calculate data vectors at requested parameters
        try:
            data_vector, _s8 = cocoa_model.calculate_data_vector(_p, return_s8=return_s8)
        except Exception as e:
            print(f'[Error] cocoa_model.calculate_data_vector fails at rank {rank}, iteration {i-rank*N_local}!')
            print(e)
            print(f'Failing at parameters:')
            for k,v in _p.items():
                print(f'  {k}: {v}')
            raise e
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
        if return_s8:
            train_sigma8_list.append(_s8)
            context = ' '.join([f'{num:e}' for num in np.hstack([params_arr, _s8, data_vector])])
            fp.write(context+"\n")
        else:
            context = ' '.join([f'{num:e}' for num in np.hstack([params_arr, data_vector])])
            fp.write(context+"\n")
        fp.flush()
    fp.close()
    if return_s8:
        return train_params_list, train_data_vector_list, train_sigma8_list
    else:
        return train_params_list, train_data_vector_list, None

def get_data_vectors(params_list, comm, rank, label, return_s8=False):
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
    local_params_list, local_data_vector_list, local_sigma8_list = get_local_data_vector_list(params_list,rank,label,return_s8=return_s8)
    comm.Barrier() # Synchronize before collecting results
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

train_samples, train_data_vectors, train_sigma8 = get_data_vectors(params_train, comm, rank, label_train, return_s8=True)
valid_samples, valid_data_vectors, valid_sigma8 = get_data_vectors(params_valid, comm, rank, label_valid, return_s8=True)

# ============ Clean training data & save ====================
if(rank==0):
    # ================== Chi_sq cut ==========================
    def get_chi_sq_cut(train_data_vectors):
        chi_sq_list = []
        for dv in train_data_vectors:
            delta_dv = (dv - config.dv_lkl)[config.mask_lkl]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
            chi_sq_list.append(chi_sq)
        chi_sq_arr = np.array(chi_sq_list)
        print(f'chi2 difference [{np.nanmin(chi_sq_arr)}, {np.nanmax(chi_sq_arr)}]')
        select_chi_sq = (chi_sq_arr < config.chi_sq_cut)
        return select_chi_sq
    # ===============================================
    select_chi_sq_train = get_chi_sq_cut(train_data_vectors)
    selected_obj_train = np.sum(select_chi_sq_train)
    total_obj_train    = len(train_data_vectors)
    print(f'[calculate_dv.py] Select {selected_obj_train} training sample out of {total_obj_train}!')
    select_chi_sq_valid = get_chi_sq_cut(valid_data_vectors)
    selected_obj_valid = np.sum(select_chi_sq_valid)
    total_obj_valid    = len(valid_data_vectors)
    print(f'[calculate_dv.py] Select {selected_obj_valid} training sample out of {total_obj_valid}!')
    # ===============================================
    train_data_vectors = train_data_vectors[select_chi_sq_train]
    train_samples      = train_samples[select_chi_sq_train]
    train_sigma8       = train_sigma8[select_chi_sq_train]
    valid_data_vectors = valid_data_vectors[select_chi_sq_valid]
    valid_samples      = valid_samples[select_chi_sq_valid]
    valid_sigma8       = valid_sigma8[select_chi_sq_valid]
    # ========================================================
    np.save(pjoin(config.traindir, f'data_vectors_{label_train}.npy'),train_data_vectors)
    np.save(pjoin(config.traindir, f'samples_{label_train}.npy'), train_samples)
    np.save(pjoin(config.traindir, f'sigma8_{label_train}.npy'), train_sigma8)
    np.save(pjoin(config.traindir, f'data_vectors_{label_valid}.npy'),valid_data_vectors)
    np.save(pjoin(config.traindir, f'samples_{label_valid}.npy'), valid_samples)
    np.save(pjoin(config.traindir, f'sigma8_{label_valid}.npy'), valid_sigma8)
    # ======================================================== 
    print(f'Done data vector calculation!')
MPI.Finalize
