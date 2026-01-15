import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5
import sys
from torchinfo import summary
from datetime import datetime

torch.autograd.set_detect_anomaly(True)
EPS = 1e-7 if torch.get_default_dtype() == torch.float32 else 1e-15

def assert_finite(x, name):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"NaN/Inf detected in {name}")
def assert_non_negative(x, name):
    if (x < 0).any():
        raise RuntimeError(f"Negative values detected in {name}")

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size, bias=False)
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2

class Better_ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(Better_ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()#torch.nn.BatchNorm1d(in_size)
        #self.norm2 = Affine()#torch.nn.BatchNorm1d(in_size)
        self.norm3 = Affine()#torch.nn.BatchNorm1d(in_size)

        self.act1 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#
        #self.act2 = #nn.Tanh()#nn.ReLU()#
        self.act3 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.layer2(o1) + xskip #(self.norm2(self.layer2(o1))) + xskip
        o3 = self.act3(self.norm3(o2))

        return o3

class ResBlock_v2(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(ResBlock_v2, self).__init__()

        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()#torch.nn.BatchNorm1d(in_size)
        self.norm2 = Affine()#torch.nn.BatchNorm1d(in_size)

        self.act1 = activation_fcn(in_size) 
        self.act2 = activation_fcn(in_size) 

        if dropout>0.0:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

    def forward(self, x):
        xskip = self.skip(x)
        o1 = self.dropout1(self.act1(self.layer1(self.norm1(x))))
        o2 = (self.dropout2(self.act2(self.layer2(self.norm2(o1))))) + xskip
        return o2
    
class ResBlock_v3(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(ResBlock_v3, self).__init__()

        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = torch.nn.LayerNorm(in_size)
        self.norm2 = torch.nn.LayerNorm(out_size)

        self.act1 = nn.SiLU() #activation_fcn(in_size) 
        self.act2 = nn.SiLU() #activation_fcn(out_size) 

        if dropout>0.0:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

    def forward(self, x):
        xskip = self.skip(x)
        o1 = self.dropout1(self.layer1(self.act1(self.norm1(x))))
        o2 = (self.dropout2(self.layer2(self.act2(self.norm2(o1))))) + xskip
        return o2

class CNNMLP(nn.Module):

    def __init__(self, input_dim, output_dim, cnn_in_channels, cnn_out_channels, cnn_kernel_size, cnn_stride, cnn_padding):

        super(CNNMLP, self).__init__()
        self.indim=input_dim
        self.outdim=output_dim
        self.in_channels=cnn_in_channels

        self.CNNtrans = nn.Linear(input_dim, output_dim)
        self.conv = nn.Conv1d(in_channels=cnn_in_channels, out_channels=cnn_out_channels, 
                              kernel_size=cnn_kernel_size, stride=cnn_stride, padding=cnn_padding)
        self.Act2 = activation_fcn(output_dim)
        #self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.CNNtrans(x)
        x = x.view(x.size(0), self.in_channels, -1)
        x = self.conv(x)
        x = x.view(x.size(0), self.outdim)
        #x = self.norm(x)
        x = self.Act2(x)
        return x

class ResBottle(nn.Module):
    def __init__(self, size, N):
        super(ResBottle, self).__init__()

        self.size = size
        self.N = N
        encoded_size = size // N

        # first layer
        self.norm1  = torch.nn.BatchNorm1d(encoded_size)
        self.layer1 = nn.Linear(size,encoded_size)
        self.act1   = nn.Tanh()

        # middle layer
        self.norm2  = torch.nn.BatchNorm1d(encoded_size)
        self.layer2 = nn.Linear(encoded_size,encoded_size)
        self.act2   = nn.Tanh()

        # last layer
        self.norm3  = torch.nn.BatchNorm1d(size)
        self.layer3 = nn.Linear(encoded_size,size)
        self.act3   = nn.Tanh()

        self.skip     = nn.Identity()#nn.Linear(size,size)
        self.act_skip = nn.Tanh()

    def forward(self, x):
        x_skip = self.act_skip(self.skip(x))

        o1 = self.act1(self.norm1(self.layer1(x)/np.sqrt(10)))
        o2 = self.act2(self.norm2(self.layer2(o1)/np.sqrt(10)))
        o3 = self.norm3(self.layer3(o2))
        o  = self.act3(o3+x_skip)

        return o

class DenseBlock(nn.Module):
    def __init__(self, size):
        super(DenseBlock, self).__init__()

        self.skip = nn.Identity()

        self.layer1 = nn.Linear(size, size)
        self.layer2 = nn.Linear(size, size)

        self.norm1 = torch.nn.BatchNorm1d(size)
        self.norm2 = torch.nn.BatchNorm1d(size)

        self.act1 = nn.Tanh()#nn.SiLU()#nn.PReLU()
        self.act2 = nn.Tanh()#nn.SiLU()#nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)
        o1    = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2    = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10)
        o     = torch.cat((o2,xskip),axis=1)
        return o

class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = activation_fcn(in_size)  #nn.Tanh()#nn.ReLU()#
        self.norm         = Affine() #torch.nn.BatchNorm1d(in_size)
        #self.act2         = nn.Tanh()#nn.ReLU()#
        #self.norm2        = torch.nn.BatchNorm1d(in_size)
        self.act3         = activation_fcn(in_size)  #nn.Tanh()
        self.norm3        = Affine() # torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights1 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights1 = nn.Parameter(weights1) # turn the weights tensor into trainable weights
        bias1 = torch.Tensor(in_size)
        self.bias1 = nn.Parameter(bias1) # turn bias tensor into trainable weights

        weights2 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights2 = nn.Parameter(weights2) # turn the weights tensor into trainable weights
        bias2 = torch.Tensor(in_size)
        self.bias2 = nn.Parameter(bias2) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights1, a=np.sqrt(5)) # matrix weights init 
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1) # fan_in in the input size, fan out is the output size but it is not use here
        bound1 = 1 / np.sqrt(fan_in1) 
        nn.init.uniform_(self.bias1, -bound1, bound1) # bias weights init

        nn.init.kaiming_uniform_(self.weights2, a=np.sqrt(5))  
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
        bound2 = 1 / np.sqrt(fan_in2) 
        nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self,x):
        mat1 = torch.block_diag(*self.weights1) # how can I do this on init rather than on each forward pass?
        mat2 = torch.block_diag(*self.weights2)
        #x_norm = self.norm(x)
        #_x = x_norm.reshape(x_norm.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        #_x = x.reshape(x.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        #o1 = self.act(self.norm(torch.matmul(x,mat1)+self.bias1))
        #o2 = torch.matmul(o1,mat2)+self.bias2  #self.act2(self.norm2(torch.matmul(o1,mat2)+self.bias2))
        #o3 = self.act3(self.norm3(o2+x))
        o1 = self.act(torch.matmul(self.norm(x),mat1)+self.bias1)
        o2 = self.act3(torch.matmul(self.norm3(o1),mat2)+self.bias2)+x
        return o2

class BlockLinear(nn.Module):
    """
    Implements a block-diagonal linear layer where each block is an nn.Linear.
    Input is split into chunks, each passed into its own linear layer.
    """
    def __init__(self, in_dims, out_dims, bias=True):
        """
        in_dims  = [d1, d2, ..., dk]
        out_dims = [o1, o2, ..., ok]
        """
        super().__init__()
        assert len(in_dims) == len(out_dims)
        
        self.blocks = nn.ModuleList([
            nn.Linear(in_d, out_d, bias=bias)
            for in_d, out_d in zip(in_dims, out_dims)
        ])
        self.in_dims = in_dims
        self.out_dims = out_dims

    def forward(self, x):
        # Split input into block segments
        xs = torch.split(x, self.in_dims, dim=-1)  # last dim split
        
        # Apply each block
        ys = [block(xi) for block, xi in zip(self.blocks, xs)]
        
        # Concatenate result to form block-diagonal output
        return torch.cat(ys, dim=-1)
class FeedForward(nn.Module):
    def __init__(self, embed_dim, block_sizes, expansion=4, dropout=0.0):
        super().__init__()
        
        in_dims  = block_sizes
        hid_dims = [d * expansion for d in block_sizes]

        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = BlockLinear(in_dims, hid_dims)
        self.fc2 = BlockLinear(hid_dims, in_dims)

        self.act = nn.SiLU()  # or SwiGLU if desired
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x + residual   # Linear skip
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_sizes, expansion=4, dropout=0.0):
        super().__init__()
        
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout_attn = nn.Dropout(dropout)
        
        self.ff = FeedForward(embed_dim, block_sizes, expansion, dropout)

    def forward(self, x):
        # ---- Attention block (PreNorm) ----
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout_attn(attn_out)   # Linear skip

        # ---- FeedForward block (PreNorm) ----
        x = self.ff(x)  # already has skip inside

        return x

class activation_fcn(nn.Module):
    ''' Trainable Swish activation function
    f(x)=(gamma+(1+exp(-beta*x))^(-1)*(1-gamma))*x
    '''
    def __init__(self, dim):
        super(activation_fcn, self).__init__()

        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))
        #self.m = nn.Sigmoid()

    def forward(self,x):
        #inv = self.m(torch.mul(self.beta,x))
        inv = torch.sigmoid(torch.mul(self.beta,x))
        fac = 1-self.gamma
        out = torch.mul(self.gamma + torch.mul(inv,fac), x)
        return out

class True_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(True_Transformer, self).__init__()  
    
        self.int_dim      = in_size//n_partitions
        self.n_partitions = n_partitions
        self.linear       = nn.Linear(self.int_dim,self.int_dim)#ResBlock(self.int_dim,self.int_dim)#
        self.act          = nn.ReLU()
        self.norm         = torch.nn.BatchNorm1d(self.int_dim*n_partitions)

    def forward(self,x):
        batchsize = x.shape[0]
        out = torch.reshape(self.norm(x),(batchsize,self.n_partitions,self.int_dim))
        out = self.act(self.linear(out))
        out = torch.reshape(out,(batchsize,self.n_partitions*self.int_dim))
        return out+x
    
class NNEmulator:
    def __init__(self, N_DIM, OUTPUT_DIM, dv_fid, dv_std, invcov, mask=None, 
        param_mask=None, model=None, deproj_PCA=False, optim=None, 
        device=torch.device('cpu'), lr=1e-3, reduce_lr=True, scheduler=None, 
        weight_decay=1e-3, dtype='float', print_summary=False, dropout=0.0):
        self.generator=torch.Generator("cpu")

        ### Set the input parameter space dimension
        self.N_DIM = N_DIM
        if param_mask is not None:
            assert len(param_mask)==N_DIM, f'Param mask size != N_DIM! {param_mask}'
            self.param_mask = np.array(param_mask).astype(bool)
            self.N_DIM_REDUCED = np.sum(param_mask)
            print(f'Only {self.N_DIM_REDUCED}/{N_DIM} params are trained on.')
        else:
            self.param_mask = np.ones(N_DIM, dtype=bool)
            self.N_DIM_REDUCED = N_DIM
        self.optim = optim
        self.deproj_PCA = deproj_PCA
        self.device = device
        self.reduce_lr = reduce_lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.trained = False

        # init data vector mask
        if mask is not None:
            self.mask = mask.astype(bool)
        else:
            self.mask = np.ones(OUTPUT_DIM, dtype=bool)
        OUTPUT_DIM_REDUCED = self.mask.sum()
        # init data vector, dv covariance, and dv std (and deproject to PCs)
        # and also get rid off masked data points
        # NOTE: double precision for these tensors to avoid numerical issues
        self.dv_fid = torch.tensor(dv_fid, dtype=torch.float64)
        self.dv_std = torch.tensor(dv_std, dtype=torch.float64)
        self.invcov = torch.tensor(invcov, dtype=torch.float64)
        if self.deproj_PCA:
            # note that mask the dv and cov before building PCs
            dv_fid_masked = torch.tensor(dv_fid[self.mask], dtype=torch.float64)
            invcov_masked = torch.tensor(invcov[self.mask][:,self.mask], dtype=torch.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(invcov_masked)
            eigenvalues = torch.as_tensor(eigenvalues, dtype=torch.float64)
            eigenvectors = torch.as_tensor(eigenvectors, dtype=torch.float64)
            assert torch.all(eigenvalues>0), 'Non-positive-def invcov!'
            self.PC_masked = eigenvectors.detach().clone()
            self.dv_std_reduced = (1./torch.sqrt(eigenvalues)).detach().clone()
            self.dv_fid_reduced = dv_fid_masked@self.PC_masked
            self.invcov_reduced = torch.diag(eigenvalues).detach().clone()
        else:
            self.PC_masked = None
            self.dv_fid_reduced = torch.tensor(dv_fid[self.mask], dtype=torch.float64)
            self.dv_std_reduced = torch.tensor(dv_std[self.mask], dtype=torch.float64)
            self.invcov_reduced = torch.tensor(invcov[self.mask][:,self.mask], dtype=torch.float64)
        
        if (model==0):
            print("Using simply connected NN...")
            self.model = nn.Sequential(
                                nn.Linear(self.N_DIM_REDUCED, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(1024, OUTPUT_DIM_REDUCED),
                                Affine()
                                )
        elif(model==1):
            print("Using resnet model...")
            self.model = nn.Sequential(
                           nn.Linear(self.N_DIM_REDUCED, 128),
                           ResBlock(128, 256),
                           ResBlock(256, 256),
                           ResBlock(256, 256),
                           ResBlock(256, 512),
                           ResBlock(512, 512),
                           ResBlock(512, 512),
                           ResBlock(512, 1024),
                           ResBlock(1024, 1024),
                           ResBlock(1024, 1024),
                           ResBlock(1024, 1024),
                           ResBlock(1024, 1024),
                           Affine(),
                           nn.PReLU(),
                           nn.Linear(1024, OUTPUT_DIM_REDUCED),
                           Affine()
                       )        
        elif(model==2):
            self.model = nn.Sequential(
                                nn.Linear(self.N_DIM_REDUCED, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, OUTPUT_DIM_REDUCED),
                                Affine()
                                )
        elif(model==3):
            self.model = nn.Sequential(
                                nn.Linear(self.N_DIM_REDUCED, 3072),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(3072, 3072),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(3072, 3072),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(3072, 3072),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(3072, OUTPUT_DIM_REDUCED),
                                Affine()
                                )
        elif(model==4):
            print("Using resnet model...")
            self.model = nn.Sequential(
                           nn.Linear(self.N_DIM_REDUCED, 1024),
                           nn.ReLU(),
                           ResBlock(1024, 512),
                           ResBlock(512, 256),
                           ResBlock(256, 128),
                           nn.Linear(128, 512),
                           nn.ReLU(),
                           nn.Linear(512, OUTPUT_DIM_REDUCED),
                           nn.ReLU(),
                           Affine()
                       )
        elif(model==5):
            print("Using Evan's ResTRF model...")
            int_dim_res = 256
            n_channels = 60
            int_dim_trf = 1024
            self.model = nn.Sequential(
                            nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            nn.Linear(int_dim_res, int_dim_trf),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            nn.Linear(int_dim_trf,OUTPUT_DIM_REDUCED),
                            Affine()
                        )
        elif(model==6):
            print("Using Evan's simplified ResNet model...")
            int_dim_res = 256
            self.model = nn.Sequential(
                            nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            nn.Linear(int_dim_res, OUTPUT_DIM_REDUCED),
                            Affine()
                        )
        elif(model==7):
            print("Using Evan's larger ResNet model...")
            int_dim_res = 1024
            self.model = nn.Sequential(
                            nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            Better_ResBlock(int_dim_res, int_dim_res),
                            nn.Linear(int_dim_res, OUTPUT_DIM_REDUCED),
                            Affine()
                        )
        elif(model==8):
            print("Using Deep ResMLP model (model 8)...")
            int_dim_res = 512
            self.model = nn.Sequential(
                            nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            nn.Linear(int_dim_res, OUTPUT_DIM_REDUCED),
                            Affine()
                        )
        elif(model==9):
            print("Using CNNMLP model (model 9)...")
            int_dim_res = 512
            cnn_dim = 5120
            cnn_in_channels = 1
            cnn_out_channels = 16
            cnn_kernel_size = 5
            cnn_stride = 16
            cnn_padding = 2
            self.model = nn.Sequential(
                nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                CNNMLP(int_dim_res, cnn_dim, cnn_in_channels, cnn_out_channels, cnn_kernel_size, cnn_stride, cnn_padding),
                nn.Linear(cnn_dim, OUTPUT_DIM_REDUCED),
                Affine()
            )
        elif(model==10):
            print("Using Resv2TRFv2 model (model 10)...")
            int_dim_res = 256
            int_dim_trf = 1024
            n_channels = 32
            #n_heads = 16
            #block_sizes = [int_dim_trf//n_heads, ] * n_heads
            #expansion = 4
            self.model = nn.Sequential(
                            nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            nn.Linear(int_dim_res, int_dim_trf),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            nn.Linear(int_dim_trf,OUTPUT_DIM_REDUCED),
                            Affine()
                        )
        elif(model==11):
            print("Using Resv2TRFv2x3 model (model 11)...")
            int_dim_res = 256
            int_dim_trf = 1024
            n_channels = 32
            #n_heads = 16
            #block_sizes = [int_dim_trf//n_heads, ] * n_heads
            #expansion = 4
            #int_dim_trf = 512
            self.model = nn.Sequential(
                            nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            nn.Linear(int_dim_res, int_dim_trf),
                            #TransformerBlock(int_dim_trf, n_heads, block_sizes, expansion=expansion, dropout=self.dropout),
                            #TransformerBlock(int_dim_trf, n_heads, block_sizes, expansion=expansion, dropout=self.dropout),
                            #TransformerBlock(int_dim_trf, n_heads, block_sizes, expansion=expansion, dropout=self.dropout),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            Better_Attention(int_dim_trf, n_channels),
                            Better_Transformer(int_dim_trf, n_channels),
                            nn.Linear(int_dim_trf,OUTPUT_DIM_REDUCED),
                            Affine()
                        )
        elif(model==12):
            print("Using ResMLP v2 model (model 12)...")
            int_dim_res = 256
            self.model = nn.Sequential(
                            nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            ResBlock_v2(int_dim_res, int_dim_res, self.dropout),
                            nn.Linear(int_dim_res, OUTPUT_DIM_REDUCED),
                            Affine()
                        )
        elif(model==13):
            print("Using CNNMLP model (model 13, with ResMLP v3)...")
            int_dim_res = 512
            cnn_dim = 5120
            cnn_in_channels = 1
            cnn_out_channels = 16
            cnn_kernel_size = 5
            cnn_stride = 16
            cnn_padding = 2
            self.model = nn.Sequential(
                nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                CNNMLP(int_dim_res, cnn_dim, cnn_in_channels, cnn_out_channels, cnn_kernel_size, cnn_stride, cnn_padding),
                nn.Linear(cnn_dim, OUTPUT_DIM_REDUCED),
                Affine()
            )
        elif(model==14):
            print("Using CNNMLP model (model 14, with ResMLP v3)...")
            int_dim_res = 512
            cnn_dim = 5120
            cnn_in_channels = 1
            cnn_out_channels = 16
            cnn_kernel_size = 5
            cnn_stride = 16
            cnn_padding = 2
            self.model = nn.Sequential(
                nn.Linear(self.N_DIM_REDUCED, int_dim_res),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                ResBlock_v3(int_dim_res, int_dim_res, self.dropout),
                CNNMLP(int_dim_res, cnn_dim, cnn_in_channels, cnn_out_channels, cnn_kernel_size, cnn_stride, cnn_padding),
                nn.Linear(cnn_dim, OUTPUT_DIM_REDUCED),
                Affine()
            )
        else:
            print(f'Can not support model {model}!')
            exit(1)
        if print_summary:
            summary(self.model)
        if dtype=='double':
            self.model = self.model.double()
            self.model_dtype = torch.double
            print('Using double precision for the model.')
        elif dtype=='float':
            self.model = self.model.float()
            self.model_dtype = torch.float32
            print('Using single precision for the model.')
        elif dtype=='half':
            self.model = self.model.half()
            self.model_dtype = torch.float16
            print('Using half precision for the model.')
        self.model.to(device)

        if self.optim is None:
            print('Learning rate = {}'.format(lr))
            print('Weight decay = {}'.format(weight_decay))
            # weight_decay used to be 1e-4 from Supranta
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr,
                weight_decay=self.weight_decay)
        # LR scheduler from Evan's emulator
        if self.reduce_lr:
            print('Reduce LR on plateau: ', self.reduce_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min', patience=15, factor=0.5)

        ### JX: Initialize model weights
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def do_pca(self, data_vector):
        assert self.deproj_PCA==True
        return data_vector@self.PC_masked
    
    def do_inverse_pca(self, PC_coeff):
        assert self.deproj_PCA==True
        return PC_coeff@(self.PC_masked.T)

    def chi2_to_loss(self, chi2_arr, loss_type="mean"):
        ''' Map dchi2 to loss functions
        '''
        assert_finite(chi2_arr, "Delta chi2 array")
        assert_non_negative(chi2_arr, "Delta chi2 array")
        if loss_type=="mean":
            loss = torch.mean(chi2_arr)
        elif loss_type=="clipped_mean":
            loss = torch.mean(torch.sort(chi2_arr)[0][:-5])
        elif loss_type=="log_chi2":
            loss = torch.mean(torch.log(chi2_arr.clamp(min=EPS)))
        elif loss_type=="log_hyperbola":
            loss = torch.mean(torch.log((1+chi2_arr).clamp(min=EPS)))
        elif loss_type=="hyperbola":
            loss = torch.mean((1+2*chi2_arr)**(1/2))-1
        elif loss_type=="hyperbola-1/3":
            loss = torch.mean((1+3*chi2_arr)**(1/3))-1
        else:
            print(f'Can not find loss function type {loss_type}!')
            print(f'Available choices: [mean, clipped_mean, log_chi2, log_hyperbola, hyperbola, hyperbola-1/3]')
            exit(1)
        assert torch.isfinite(loss), f'Invalid loss: {chi2_arr.detach().cpu()}'
        return loss

    def monitor_grad(self, epoch):
        ''' Monitor the model parameters gradient for debugging purpose
        '''
        NaN_norm_counts = 0
        min_norm, max_norm = np.inf, -np.inf
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                if torch.isfinite(param_norm):
                    min_norm = min_norm if param_norm > min_norm else param_norm
                    max_norm = max_norm if param_norm < max_norm else param_norm
                else:
                    NaN_norm_counts += 1
        print(f'\rEpoch {epoch:3d}: Gradient norm range [{min_norm:.1e}, {max_norm:.1e}], {NaN_norm_counts:d} NaN grad', 
              end='', flush=True)

    def train(self, X, y, X_validation, y_validation, batch_size=1000, 
        n_epochs=150, loss_type="mean", debug_grad=False, save_loss_filename=None):
        ''' Train the network with data vectors normalized by covariance PCs
        y: training/validation data
        Y: model prediction
        X: input parameters
        '''
        print('Start emulator training...')
        print('Batch size = ',batch_size)
        print('N_epochs = ',n_epochs)
        print(f'PC projection = {self.deproj_PCA}')
        print(f'Parameters after masking = {self.N_DIM_REDUCED}')
        print(f'Data vector points after masking = {self.mask.sum()}')
        # pre-process training & validation data sets
        if self.deproj_PCA:
            # reduce & normalize y and y_validation by PC, and subtract mean
            y_reduced = self.do_pca(y[:,self.mask]) - self.dv_fid_reduced
            y_validation_reduced = self.do_pca(y_validation[:,self.mask]) - self.dv_fid_reduced
        else:
            # only get rid off masked elements
            y_reduced = (y[:,self.mask]).double() - self.dv_fid_reduced
            y_validation_reduced = (y_validation[:,self.mask]).double() - self.dv_fid_reduced
        X_reduced = (X[:,self.param_mask]).double()
        X_validation_reduced = (X_validation[:,self.param_mask]).double()

        # get normalization factors, float64 for better dynamic range
        if not self.trained:
            self.X_mean = X_reduced.mean(axis=0, keepdims=True).clone().detach()
            self.X_std  = X_reduced.std(axis=0, keepdims=True).clone().detach()
            self.y_mean = self.dv_fid_reduced.clone().detach()
            self.y_std  = self.dv_std_reduced.clone().detach()
        # initialize arrays
        losses_train = []
        losses_vali = []
        loss = 100.

        # send everything to device
        self.model.to(self.device)
        tmp_y_std        = self.y_std.to(self.device)
        tmp_cov_inv      = self.invcov_reduced.to(self.device)
        tmp_X_mean       = self.X_mean.to(self.device)
        tmp_X_std        = self.X_std.to(self.device)
        tmp_X_validation = ((X_validation_reduced.to(self.device) - tmp_X_mean)/tmp_X_std).to(self.model_dtype)
        tmp_y_validation = y_validation_reduced.to(self.device)

        # Normalize in float64, then convert to float32 for model input
        X_train     = ((X_reduced - self.X_mean)/self.X_std).to(self.model_dtype)
        y_train     = y_reduced
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        validset    = torch.utils.data.TensorDataset(tmp_X_validation,tmp_y_validation)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, 
                                                  drop_last=True, num_workers=0, generator=self.generator)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                                  drop_last=True, num_workers=0, generator=self.generator)

        print('Datasets loaded!')
        print('Begin training...')
        if save_loss_filename is not None:
            loss_fp = open(save_loss_filename, 'w')
            loss_fp.write('# epoch   train_loss   valid_loss\n')
        train_start_time = datetime.now()
        for e in range(n_epochs):
            start_time = datetime.now()

            # training loss
            self.model.train()
            losses = []
            for data in trainloader:
                # normalized X and y
                X       = data[0].to(self.device) # self.model_dtype
                y_batch = data[1].to(self.device) # float64
                # make sure dchi2 calculation in float64
                Y_pred  = self.model(X).to(torch.float64) * tmp_y_std
                assert_finite(Y_pred, "Model prediction")
                diff = (y_batch - Y_pred).to(torch.float64)
                chi2_arr = torch.diag((diff@tmp_cov_inv)@torch.t(diff))
                loss = self.chi2_to_loss(chi2_arr, loss_type)
                losses.append(loss.cpu().detach().numpy())
                self.optim.zero_grad()
                loss.backward()
                if debug_grad:
                    self.monitor_grad(e)
                # clipping exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
                self.optim.step()
            losses_train.append(np.mean(losses))

            # validation loss
            with torch.no_grad():
                self.model.eval()
                losses = []
                for data in validloader:  
                    X_v       = data[0].to(self.device) # self.model_dtype
                    y_v_batch = data[1].to(self.device) # float64
                    # make sure dchi2 calculation in float64
                    Y_v_pred = self.model(X_v).to(torch.float64) * tmp_y_std
                    assert_finite(Y_v_pred, "Model prediction")
                    v_diff = (y_v_batch - Y_v_pred).to(torch.float64)
                    chi2_arr_v = torch.diag((v_diff@tmp_cov_inv)@torch.t(v_diff))
                    loss_v = self.chi2_to_loss(chi2_arr_v, loss_type)
                    losses.append(loss_v.cpu().detach().numpy())
                losses_vali.append(np.mean(losses))
                if self.reduce_lr:
                    self.scheduler.step(losses_vali[e])
                self.optim.zero_grad()
            # count per epoch time consumed 
            end_time = datetime.now()
            epoch_cost = (end_time-start_time).total_seconds()
            print(f'\nEpoch {e:3d}: {epoch_cost:.2f} s; lr = {self.optim.param_groups[0]["lr"]:.2e}')
            print(f'--- Training loss = <{losses_train[-1]:.2e}>')
            print(f'--- Validation loss = <<{losses_vali[-1]:.2e}>>')
            if save_loss_filename is not None:
                loss_fp.write(f'{e:3d} {losses_train[-1]:.6e} {losses_vali[-1]:.6e}\n')
                loss_fp.flush()
        # Finish all the epochs
        if save_loss_filename is not None:
            loss_fp.close()
        train_end_time = datetime.now()
        train_cost = (train_end_time-train_start_time).total_seconds()/3600.
        print(f'Training cost: {train_cost:.2} hours')
        self.trained = True

    def predict(self, X):
        ''' Predict unmasked data vector based on input parameters (tensor)
        '''
        assert self.trained, "The emulator needs to be trained first before predicting"
        assert X.dtype==torch.get_default_dtype()
        # wrap the input X if it's 1D
        if X.dim()==1:
            X = torch.atleast_2d(X)
        elif X.dim()>2:
            print(f'Error: Can not support {X.dim()}-dimension input X!')
            exit(1)

        # do prediction from 2D X
        with torch.no_grad():
            self.model.eval()
            X_mean = self.X_mean.clone().detach()
            X_std  = self.X_std.clone().detach()
            y_mean = self.y_mean.clone().detach()
            y_std  = self.y_std.clone().detach()

            X_norm = ((X[:,self.param_mask].clone().detach().to(torch.float64) - X_mean) / X_std).to(self.model_dtype)
            y_pred = (self.model(X_norm).cpu()).to(torch.float64) * y_std + y_mean
        if self.deproj_PCA:
            data_vector_masked = self.do_inverse_pca(y_pred).numpy()
        else:
            data_vector_masked = y_pred.numpy()
        data_vector = np.zeros([X.size(0), self.mask.shape[0]])
        data_vector[:,self.mask] = data_vector_masked
        return data_vector
        
    def save(self, filename):
        torch.save(self.model, filename)
        with h5.File(filename + '.h5', 'w') as f:
            # TODO: what data to save?
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['Y_mean'] = self.y_mean
            f['Y_std']  = self.y_std
            # f['dv_fid'] = self.dv_fid
            # f['dv_std'] = self.dv_std
            # f['dv_fid_reduced'] = self.dv_fid_reduced
            # f['dv_std_reduced'] = self.dv_std_reduced
            f['mask'] = self.mask
            f['param_mask'] = self.param_mask
            f['deproj_PCA'] = self.deproj_PCA
            f['model_dtype'] = str(self.model_dtype)
            if self.deproj_PCA:
                f['PC_masked'] = self.PC_masked
        
    def load(self, filename, device=torch.device('cpu'),state_dict=False):
        self.trained = True
        #if device!=torch.device('cpu'):
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')
        if state_dict==False:
            self.model = torch.load(filename,map_location=device)
        else:
            print('Loading with "torch.load_state_dict(torch.load(file))"...')
            self.model.load_state_dict(torch.load(filename,map_location=device))
        self.model.eval()
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.tensor(f['X_mean'][:], dtype=torch.float64)
            self.X_std  = torch.tensor(f['X_std'][:], dtype=torch.float64)
            self.y_mean = torch.tensor(f['Y_mean'][:], dtype=torch.float64)
            self.y_std  = torch.tensor(f['Y_std'][:], dtype=torch.float64)
            self.mask = f['mask'][:].astype(bool)
            self.param_mask = f['param_mask'][:].astype(bool)
            self.deproj_PCA = f['deproj_PCA'][()].astype(bool)
            try:
                self.model_dtype = eval(f['model_dtype'][()].decode('utf-8'))
            except:
                self.model_dtype = torch.float64
            if self.deproj_PCA:
                self.PC_masked = torch.tensor(f['PC_masked'][:], dtype=torch.float64)
                self.invcov_reduced = torch.diag(1./self.y_std**2)
            else:
                self.PC_masked = None