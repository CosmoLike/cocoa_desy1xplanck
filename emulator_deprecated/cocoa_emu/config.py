import yaml
import numpy as np
import os
from os.path import join as pjoin
# from .sampling import get_starting_pos
import copy
from .utils import readDatasetFile

class Config:
    # multi-probe mask, in sequence of xipm, gammat, wtheta, wgk, wsk, ckk
    # valid probes:
    # xi, wtheta, gammat, 2x2pt, 3x2pt, xi_ggl, 5x2pt, 6x2pt, c3x2pt
    probe_mask_choices = {
        "xi": [1, 0, 0, 0, 0, 0],
        "wtheta":       [0, 0, 1, 0, 0, 0],
        "gammat":       [0, 1, 0, 0, 0, 0],
        "2x2pt":        [0, 1, 1, 0, 0, 0],
        "3x2pt":        [1, 1, 1, 0, 0, 0],
        "xi_ggl":       [1, 1, 0, 0, 0, 0],
        "5x2pt":        [1, 1, 1, 1, 1, 0],
        "c3x2pt":       [0, 0, 0, 1, 1, 1],
        "6x2pt":        [1, 1, 1, 1, 1, 1],
        "CMBL":         [0, 0, 0, 0, 0, 1],
        "gk2x2pt":      [0, 0, 1, 1, 0, 0],
        "sk2x2pt":      [1, 0, 0, 0, 1, 0],
        "gk3x2pt":      [0, 0, 1, 1, 0, 1],
        "sk3x2pt":      [1, 0, 0, 0, 1, 1],
    }

    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        
        self.config_args_emu = config_args['emulator'] 
        self.params = config_args['params'] 
        config_args_lkl = config_args['likelihood']
        
        self.load_params(self.params)     
        self.load_lkl(config_args_lkl)
        self.load_emu(self.config_args_emu)

    def load_lkl(self, config_args_lkl):
        ''' Setup likelihood related datasets
        Input:
        ======
            - config_args_lkl: the `likelihood` section in the YAML file, dict
        '''
        self.likelihood = list(config_args_lkl.keys())[0]
        self.probe = self.likelihood[self.likelihood.find('_')+1:]
        self.probe_mask = self.probe_mask_choices[self.probe]
        self.config_args_lkl = config_args_lkl[self.likelihood]

        dataset = readDatasetFile(self.config_args_lkl['data_file'], 
            root=self.config_args_lkl['path'])
        dst = pjoin(self.config_args_lkl['path'], "datasets")

        # Read data vector & tomography dimension
        self.source_ntomo = int(dataset.get("source_ntomo", 0))
        self.lens_ntomo = int(dataset.get("lens_ntomo", 0))
        self.Ntheta = int(dataset.get("n_theta", 0))
        self.Nbp = int(dataset.get("n_bp", 0))
        self.Nell = int(dataset.get("n_ell", 0))
        self.lensing_overlap_cut = float(dataset.get("lensing_overlap_cut", 0.))
        assert np.abs(self.lensing_overlap_cut)<1e-5, "ERROR: The emulator only supports lensing_overlap_cut = 0.0 for now!"
        self.probe_size = [
            int(self.source_ntomo*(self.source_ntomo+1)*self.Ntheta),
            self.source_ntomo*self.lens_ntomo*self.Ntheta,
            self.lens_ntomo*self.Ntheta,
            self.lens_ntomo*self.Ntheta,
            self.source_ntomo*self.Ntheta,
            self.Nbp]

        # Read mask, data vector, and baryon feedback PCs
        self.mask_lkl = np.loadtxt(pjoin(dst, dataset["mask_file"]))[:,1].astype(bool)
        self.dv_lkl = np.loadtxt(pjoin(dst, dataset["data_file"]))[:,1]
        try:
            self.baryon_pcas = np.loadtxt(pjoin(dst,dataset["baryon_pca_file"]))
        except:
            self.baryon_pcas = None

        # Read covariance and point-mass correction -> inv cov
        invcov = self.get_full_cov(pjoin(dst, dataset["cov_file"]))
        self.dv_std = np.sqrt(np.diagonal(invcov))
        # Add Hartlap factor to CMB lensing covariance
        self.Hartlap = 1 if "Hartlap_Nvar" not in dataset else int(dataset["Hartlap_Nvar"])
        if self.Hartlap>1:
            self.Hartlap = (self.Hartlap - self.Nbp -2.0)/(self.Hartlap - 1.0)
        invcov[-self.Nbp:,-self.Nbp:] /= self.Hartlap
        invcov = np.linalg.inv(invcov[self.mask_lkl][:,self.mask_lkl])
        # Add PM marginalization
        if "U_PMmarg" in dataset:
            U_PMmarg = np.loadtxt(pjoin(dst, dataset["U_PMmarg"]))
            U = np.zeros([self.mask_lkl.shape[0], self.lens_ntomo], dtype=np.float64)
            for line in U_PMmarg:
                i, j = int(line[0]), int(line[1])
                U[i,j] = float(line[2])
            U = U[self.mask_lkl,:]
            central_block = np.diag(np.ones(self.lens_ntomo, dtype=np.float64)) + U.T@invcov@U
            w, v = np.linalg.eig(central_block)
            assert np.min(w)>=0, f'Central block not positive-definite!'
            corr = invcov @ (U@np.linalg.inv(central_block)@U.T) @ invcov
            invcov -= corr
        self.masked_inv_cov = invcov
        # test positive-definite; compare accu between Python v.s. C++ PMmarg
        w, v = np.linalg.eig(self.masked_inv_cov)
        assert np.min(w)>=0, f'Precision matrix not positive-definite after PMmarg!'
        self.inv_cov = np.zeros([self.mask_lkl.shape[0],self.mask_lkl.shape[0]], dtype=np.float64)
        for i in range(self.inv_cov.shape[0]):
            for j in range(self.inv_cov.shape[1]):
                if (self.mask_lkl[i]>0) and (self.mask_lkl[j]>0):
                    i_reduce, j_reduce = int(self.mask_lkl[:i].sum()), int(self.mask_lkl[:j].sum())
                    self.inv_cov[i,j] = self.masked_inv_cov[i_reduce,j_reduce]

    def load_emu(self, config_args_emu):
        # Read emulator related data
        # self.mask_emu = np.loadtxt(config_args_emu['sampling']['scalecut_mask'])[:,1].astype(bool)
        # self.dv_fid = np.loadtxt(config_args_emu['training']['dv_fid'])[:,1]
        # also train an emulator for sigma_8 at z=0
        self.derived = 1
        self.sigma8_fid = np.array([config_args_emu['derived']['sigma8_fid']])
        self.sigma8_std = np.array([config_args_emu['derived']['sigma8_std']])
        try:
            self.n_pcas_baryon = config_args_emu['baryons']['n_pcas_baryon']
        except:
            self.n_pcas_baryon = 0
        try:
            self.chi_sq_cut = config_args_emu['training']['chi_sq_cut']
        except:
            self.chi_sq_cut = 1e+5

        self.shear_calib_mask = np.load(config_args_emu['shear_calib']['mask'])
        # Fast parameters sequence: shear calibration, baryon PCs
        # Note: 1. Those fast parameters are not being sampled in the YAML file
        #       2. Technically linear galaxy bias is not a fast parameter due to
        #          RSD and magnification bias.
        #       3. We are including linear gbias as slow parameters now.
        # if self.probe != 'xi':
        #     self.galaxy_bias_mask = np.load(self.config_args_emu['galaxy_bias']['mask'])
        #     self.n_fast_pars = self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo
        # else:
        #     self.n_fast_pars = self.n_pcas_baryon + self.source_ntomo
        self.n_fast_pars = self.source_ntomo + self.n_pcas_baryon
        # assert len(self.dv_lkl)==len(self.dv_fid),"Observed data vector is of different size compared to the fiducial data vector."

        # Set I/O path
        self.savedir = config_args_emu['io']['savedir']
        os.makedirs(self.savedir, exist_ok=True)
        try:
            self.chaindir = config_args_emu['io']['chaindir']
        except:
            self.chaindir = os.path.join(self.savedir, "validating_chains")
        os.makedirs(self.chaindir, exist_ok=True)
        self.traindir = os.path.join(self.savedir, "training_sample")
        self.modeldir = os.path.join(self.savedir, "model_dataset")
        os.makedirs(self.traindir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)
        try:
            self.chainname = config_args_emu['io']['chainname']
        except:
            self.chainname = 'emu_chain'
        try:
            self.save_train_data = config_args_emu['io']['save_train_data']
        except:
            self.save_train_data = False
        try:
            self.save_intermediate_model = config_args_emu['io']['save_intermediate_model']
        except:
            self.save_intermediate_model = False

        # Read emulator architecture
        self.emu_type = config_args_emu['training']['emu_type']
        self.loss_type = config_args_emu['training']['loss_type']
        self.learning_rate = config_args_emu['training']['learning_rate']
        self.weight_decay = config_args_emu['training']['weight_decay']
        self.reduce_lr = config_args_emu['training']['reduce_lr']
        self.do_pca = config_args_emu['training']['do_pca']
        self.dropout = config_args_emu['training']['dropout']
        assert (self.emu_type.lower()=='nn') or (self.emu_type.lower()=='gp'),\
                        "emu_type has to be either gp or nn."
        if(self.emu_type.lower()=='nn'):
            self.batch_size = int(config_args_emu['training']['batch_size'])
            self.n_epochs = int(config_args_emu['training']['n_epochs'])
            try:
                self.nn_model  = int(config_args_emu['training']['nn_model'])
            except:
                self.nn_model  = 0
        elif(self.emu_type.lower()=='gp'):
            self.gp_resample   = int(config_args_emu['training']['gp_resample'])

        # Read training sample settings
        _init_sample = config_args_emu['init_sample']
        self.init_sample_type = _init_sample["type"]
        if self.init_sample_type == "lhs":
            self.n_lhs = int(_init_sample['lhs_n'])
            self.lhs_minmax = self.get_lhs_minmax()
        elif self.init_sample_type == "gaussian":
            self.gauss_cov = _init_sample['gauss_cov']
            self.gtemp_t = _init_sample.get('gauss_temp_train', 1.)
            self.gshift_t = _init_sample.get('gauss_shift_train', None)
            self.gnsamp_t = _init_sample.get('n_train')
            self.gtemp_v = _init_sample.get('gauss_temp_valid', 1.)
            self.gshift_v = _init_sample.get('gauss_shift_valid', None)
            self.gnsamp_v = _init_sample.get('n_valid')
            self.gauss_minmax = self.get_gaussian_minmax()
        else:
            print(f'Can not recognize init sample type {self.init_sample_type}')
            exit(1)
        self.n_train_iter = int(config_args_emu['training']['n_train_iter'])

        # Read the emcee sampler setting
        self.n_emcee_walkers=int(config_args_emu['sampling']['n_emcee_walkers'])
        self.n_mcmc = int(config_args_emu['sampling']['n_mcmc'])
        self.n_burn_in = int(config_args_emu['sampling']['n_burn_in'])
        self.n_thin = int(config_args_emu['sampling']['n_thin'])
        
        # Read parameter blocking settings
        try:
            _args_block = config_args_emu['sampling']['params_blocking']
            self.block_bias        = _args_block.get('block_bias', False)
            self.block_shear_calib = _args_block.get('block_shear_calib', False)
            self.block_dz          = _args_block.get('block_dz', False)
            self.block_ia          = _args_block.get('block_ia', False)
        except:
            self.block_bias        = False
            self.block_shear_calib = False
            self.block_dz          = False
            self.block_ia          = False
        try:
            block_label = config_args_emu['sampling']['params_blocking']['block_label'].split(',')
            block_value = config_args_emu['sampling']['params_blocking']['block_value'].split(',')
            block_value = [float(val) for val in block_value]
            block_indices = []
            for label in block_label:
                for i, param_label in enumerate(self.running_params):
                    if(label==param_label):
                        block_indices.append(i)
            self.block_indices = block_indices
            self.block_value   = block_value
        except:
            self.block_indices = []
            self.block_value   = None

        # Read debug outputs
        try:
            self.test_sample_file = config_args_emu['test']['test_samples']
            self.test_output_file = config_args_emu['test']['test_output']
        except:
            self.test_sample_file = None
            self.test_output_file = None

    def load_params(self, param_args):
        ''' Initialize likelihood model parameter settings
        TODO:
            - add fast parameter settings here?
        '''
        params_list = param_args.keys()

        self.running_params       = []
        self.running_params_type  = [] # 1:cosmo 2:src nui 3:lens nui
        self.running_params_latex = []
        self.running_params_fid   = []
        self.running_params_min   = []
        self.running_params_max   = []

        for param in params_list:
            keys = param_args[param].keys()
            # if the parameter is being sampled as recorded in the YAML
            if('value' not in keys and 'derived' not in keys and len(keys)>1):
                _args = param_args[param]
                self.running_params.append(param)
                self.running_params_latex.append(_args['latex'])
                # set the parameter boundary
                if (_args["prior"].get("dist", "uniform")=="uniform"):
                    self.running_params_fid.append(_args["ref"]["loc"])
                    self.running_params_min.append(_args["prior"]["min"])
                    self.running_params_max.append(_args["prior"]["max"])
                else:
                    self.running_params_fid.append(_args["prior"]["loc"])
                    self.running_params_min.append(-np.inf)
                    self.running_params_max.append(np.inf)
                # determine if the param is cosmological or nuisance
                if param.startswith("DES")==False:
                    self.running_params_type.append(1) # cosmology parameters
                elif param.startswith("DES_A") or param.startswith("DES_BTA"):
                    self.running_params_type.append(2) # IA parameters, src
                elif param.startswith("DES_DZ_S"):
                    self.running_params_type.append(2) # Source photo-z, src
                elif param.startswith("DES_DZ_L"):
                    self.running_params_type.append(3) # Lens photo-z, lens
                elif param.startswith("DES_STRETCH_L"):
                    self.running_params_type.append(3) # Lens photo-z stretch, lens
                elif param.startswith("DES_B1"):
                    self.running_params_type.append(3) # Lens linear gbias, lens
                else:
                    print(f'[config.py:Config.load_params]: Can not support param {param} now!')
                    exit(1)
        self.n_dim = len(self.running_params) # total param counts w/o fast ones
        self.n_pars_cosmo = self.get_Npars_cosmo()
        # params mask for each probe in [xipm, gammat, wtheta, wgk, wsk, Ckk]
        self.running_params_type = np.array(self.running_params_type)
        self.probe_params_mask = [
            (self.running_params_type==1)|(self.running_params_type==2), # xipm
            (self.running_params_type==1)|(self.running_params_type==2)|(self.running_params_type==3), # gammat
            (self.running_params_type==1)|(self.running_params_type==3), # wtheta
            (self.running_params_type==1)|(self.running_params_type==3), # wgk
            (self.running_params_type==1)|(self.running_params_type==2), # wsk
            (self.running_params_type==1), # Ckk
        ]

        return


    def get_lhs_minmax(self):
        lh_minmax = {}
        for x in self.params:
            if('prior' in self.params[x]):
                prior = self.params[x]['prior']
                if('dist' in prior):
                    loc   = prior['loc']
                    scale = prior['scale']
                    lh_min = loc - 4. * scale
                    lh_max = loc + 4. * scale
                else:
                    lh_min = prior['min']
                    lh_max = prior['max']
                lh_minmax[x] = {'min': lh_min, 'max': lh_max}
        return lh_minmax

    def get_gaussian_minmax(self):
        gauss_minmax = {}
        for x in self.params:
            if('prior' in self.params[x]):
                prior = self.params[x]['prior']
                dist = prior.get("dist", "uniform")
                if dist=="norm":
                    gauss_min = -np.inf
                    gauss_max = np.inf
                else:
                    gauss_min = prior['min']
                    gauss_max = prior['max']
                gauss_minmax[x] = {'min': gauss_min, 'max': gauss_max}
        return gauss_minmax
    
    def get_full_cov(self, cov_file):
        full_cov = np.loadtxt(cov_file)
        Ndim = len(self.dv_lkl)
        cov = np.zeros((Ndim, Ndim), dtype=np.float64)
        cov_scenario = full_cov.shape[1]
        
        for line in full_cov:
            i = int(line[0])
            j = int(line[1])

            if(cov_scenario==3):
                cov_ij = line[2]
            elif(cov_scenario==10):
                cov_g_block  = line[8]
                cov_ng_block = line[9]
                cov_ij = cov_g_block + cov_ng_block

            cov[i,j] = cov_ij
            cov[j,i] = cov_ij

        return cov
    
    def get_Npars_cosmo(self):
        Npars_cosmo = 0
        for x in self.params:
            if('prior' in self.params[x]) and (x[:3]!="DES"):
                Npars_cosmo += 1
        return Npars_cosmo
