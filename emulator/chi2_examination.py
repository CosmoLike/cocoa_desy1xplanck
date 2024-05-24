import sys
from os.path import join as pjoin
from mpi4py import MPI
import numpy as np
from cocoa_emu import Config, get_params_list, CocoaModel

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

configfile = sys.argv[1]
eval_samples_fn = sys.argv[2]
##data_vector_fn = sys.argv[3]


#config = Config(configfile)
cocoa_model = CocoaModel(configfile, "desy1xplanck.desy3xplanck_3x2pt")

def get_local_data_vector_list(params_list, rank):
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
    data_vector_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        # Here it calls cocoa to calculate data vectors at requested parameters
        data_vector = cocoa_model.calculate_data_vector(params_list[i], return_s8=False)
        data_vector_list.append(data_vector)
        
    return data_vector_list

def get_data_vectors(params_list, comm, rank):
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
    local_data_vector_list = get_local_data_vector_list(params_list, rank)
    if rank!=0:
        comm.send(local_data_vector_list, dest=0)
        data_vectors = None
    else:
        data_vector_list = local_data_vector_list
        for source in range(1,size):
            new_data_vector_list = comm.recv(source=source)
            data_vector_list = data_vector_list + new_data_vector_list
        data_vectors = np.vstack(data_vector_list)
    return data_vectors

eval_samples = np.load(eval_samples_fn)
eval_data_vectors = get_data_vectors(eval_samples, comm, rank)

np.save("/groups/timeifler/jiachuanxu/cocoa_chains/ccc/test_model_vectors.npy",
	eval_data_vectors)
