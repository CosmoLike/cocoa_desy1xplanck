import yaml
import numpy as np
# from .sampling import get_starting_pos

class Config:
    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        
        self.config_args_emu = config_args['emulator'] 
        self.params          = config_args['params'] 
        config_args_lkl = config_args['likelihood']
        
        # valid probes:
        # xi, wtheta, gammat, 2x2pt, 3x2pt, xi_ggl, 5x2pt, 6x2pt, c3x2pt
        self.shear_shear = 0
        self.shear_pos = 0
        self.pos_pos = 0
        self.gk = 0
        self.ks = 0
        self.kk = 0
        self.probe = self.config_args_emu['probe']
        if (self.probe=='cosmic_shear'):
            self.shear_shear = 1
        elif (self.probe=='wtheta'):
            self.pos_pos = 1
        elif (self.probe=='gammat'):
            self.shear_pos = 1
        elif (self.probe=='2x2pt'):
            self.shear_pos = 1
            self.pos_pos = 1
        elif (self.probe=='3x2pt'):
            self.shear_shear = 1
            self.shear_pos = 1
            self.pos_pos = 1
        elif (self.probe=='xi_ggl'):
            self.shear_shear = 1
            self.shear_pos = 1
        elif (self.probe=='5x2pt'):
            self.shear_shear = 1
            self.shear_pos = 1
            self.pos_pos = 1
            self.gk = 1
            self.ks = 1
        elif (self.probe=='c3x2pt'):
            self.gk = 1
            self.ks = 1
            self.kk = 1
        elif (self.probe=='6x2pt'):
            self.shear_shear = 1
            self.shear_pos = 1
            self.pos_pos = 1
            self.gk = 1
            self.ks = 1
            self.kk = 1
        else:
            print(f'Probe {self.probe} is not supported!')
            exit(-1)
        self.savedir   = self.config_args_emu['io']['savedir']
        try:
            self.chaindir = self.config_args_emu['io']['chaindir']
        except:
            self.chaindir = self.config_args_emu['io']['savedir']
        try:
            self.chainname = self.config_args_emu['io']['chainname']
        except:
            self.chainname = 'emu_chain'
        try:
            self.save_train_data = self.config_args_emu['io']['save_train_data']
        except:
            self.save_train_data = False
        try:
            self.save_intermediate_model = self.config_args_emu['io']['save_intermediate_model']
        except:
            self.save_intermediate_model = False
        try:
            self.n_pcas_baryon = self.config_args_emu['baryons']['n_pcas_baryon']
        except:
            self.n_pcas_baryon = 0

        try:
            self.chi_sq_cut    = float(self.config_args_emu['training']['chi_sq_cut'])
        except:
            self.chi_sq_cut = 1e+5
        # also train an emulator for sigma_8 at z=0
        try:
            self.derived = 1
            self.sigma8_fid = float(self.config_args_emu['derived']['sigma8_fid'])
            self.sigma8_std = float(self.config_args_emu['derived']['sigma8_std'])
        except:
            self.derived = 0
        
        self.dv_fid_path   = self.config_args_emu['training']['dv_fid']
        self.n_lhs         = int(self.config_args_emu['training']['n_lhs'])
        self.n_train_iter  = int(self.config_args_emu['training']['n_train_iter'])
        self.n_resample    = int(self.config_args_emu['training']['n_resample'])
        self.emu_type      = self.config_args_emu['training']['emu_type']
        assert (self.emu_type.lower()=='nn') or (self.emu_type.lower()=='gp'),\
                        "emu_type has to be either gp or nn."
        if(self.emu_type.lower()=='nn'):
            self.batch_size    = int(self.config_args_emu['training']['batch_size'])
            self.n_epochs      = int(self.config_args_emu['training']['n_epochs'])
            try:
                self.nn_model  = int(self.config_args_emu['training']['nn_model'])
            except:
                self.nn_model  = 0
        elif(self.emu_type.lower()=='gp'):
            self.gp_resample   = int(self.config_args_emu['training']['gp_resample'])
                        
        self.config_data(config_args_lkl)
        
        self.n_emcee_walkers = int(self.config_args_emu['sampling']['n_emcee_walkers'])
        self.n_mcmc          = int(self.config_args_emu['sampling']['n_mcmc'])
        self.n_burn_in       = int(self.config_args_emu['sampling']['n_burn_in'])
        self.n_thin          = int(self.config_args_emu['sampling']['n_thin'])
        self.temper0          = float(self.config_args_emu['sampling']['temper0'])
        self.temper_increment = float(self.config_args_emu['sampling']['temper_increment'])
        
        self.lhs_minmax    = self.get_lhs_minmax()
        self.n_dim         = len(self.lhs_minmax)
        self.n_pars_cosmo  = self.get_Npars_cosmo()
        
        self.param_labels = list(self.lhs_minmax.keys())
        try:
            self.block_bias        = self.config_args_emu['sampling']['params_blocking']['block_bias']
            self.block_shear_calib = self.config_args_emu['sampling']['params_blocking']['block_shear_calib']
            self.block_dz          = self.config_args_emu['sampling']['params_blocking']['block_dz']
            self.block_ia          = self.config_args_emu['sampling']['params_blocking']['block_ia']
        except:
            self.block_bias        = False
            self.block_shear_calib = False
            self.block_dz          = False
            self.block_ia          = False
        try:
            block_label = self.config_args_emu['sampling']['params_blocking']['block_label'].split(',')
            block_value = self.config_args_emu['sampling']['params_blocking']['block_value'].split(',')
            block_value = [float(val) for val in block_value]
            block_indices = []
            for label in block_label:
                for i, param_label in enumerate(self.param_labels):
                    if(label==param_label):
                        block_indices.append(i)
            self.block_indices = block_indices
            self.block_value   = block_value
        except:
            self.block_indices = []
            self.block_value   = None
        try:
            self.test_sample_file = self.config_args_emu['test']['test_samples']
            self.test_output_file = self.config_args_emu['test']['test_output']
        except:
            self.test_sample_file = None
            self.test_output_file = None
        
    def config_data(self, config_args_lkl):
        self.likelihood      = list(config_args_lkl.keys())[0]
        self.config_args_lkl = config_args_lkl[self.likelihood]
        self.likelihood_path = self.config_args_lkl['path']
        self.dataset_path    = self.likelihood_path + '/' + self.config_args_lkl['data_file']

        # Read the cocoa project dataset file
        self.Nell = 0
        self.Ntheta = 0
        self.Nbp = 0
        self.lensing_overlap_cut = 0
        self.Hartlap = 1
        with open(self.dataset_path, 'r') as f:
            for line in f.readlines():
                split_line = line.split()
                if(len(split_line)>0):
                    if(split_line[0]=='mask_file'):
                        self.mask_ones_path = self.likelihood_path + '/' + split_line[-1]
                    if(split_line[0]=='data_file'):
                        self.dv_obs_path = self.likelihood_path + '/' + split_line[-1]
                    if(split_line[0]=='cov_file'):
                        cov_file        = self.likelihood_path + '/' + split_line[-1]
                    if(split_line[0]=='baryon_pca_file'):
                        baryon_pca_file = self.likelihood_path + '/' + split_line[-1]
                    if(split_line[0]=='source_ntomo'):
                        self.source_ntomo = int(split_line[-1])
                    if self.probe != 'cosmic_shear':
                        if(split_line[0]=='lens_ntomo'):
                            self.lens_ntomo = int(split_line[-1])
                    if(split_line[0]=='n_theta'):
                        self.Ntheta = int(split_line[-1])
                    if(split_line[0]=='n_bp'):
                        self.Nbp = int(split_line[-1])
                    if(split_line[0]=='n_ell'):
                        self.Nell = int(split_line[-1])
                    if(split_line[0]=='lensing_overlap_cut'):
                        self.lensing_overlap_cut = float(split_line[-1])
                    if(split_line[0]=='Hartlap_Nvar'):
                        self.Hartlap = int(split_line[-1])
        assert np.abs(self.lensing_overlap_cut)<1e-5, "ERROR: The emulator only supports lensing_overlap_cut = 0.0 for now!"
        if self.Hartlap>1:
            self.Hartlap = (self.Hartlap - self.Nbp -2.0)/(self.Hartlap - 1.0)
        # Init data vector dimension of each components
        print("Initializing configuration space data vector dimension!")
        self.N_xi = int(self.source_ntomo*(self.source_ntomo+1)/2*self.Ntheta)
        self.N_ggl= int(self.source_ntomo*self.lens_ntomo*self.Ntheta)
        self.N_w  = int(self.lens_ntomo*self.Ntheta)
        self.N_gk = int(self.lens_ntomo*self.Ntheta)
        self.N_sk = int(self.source_ntomo*self.Ntheta)
        self.N_kk = int(self.Nbp)
        print("N_xip: %d"%(self.N_xi))
        print("N_xim: %d"%(self.N_xi))
        print("N_ggl: %d"%(self.N_ggl))
        print("N_w: %d"%(self.N_w))
        print("N_gk: %d"%(self.N_gk))
        print("N_sk: %d"%(self.N_sk))
        print("N_kk: %d"%(self.N_kk))
        try:
            self.baryon_pcas = np.loadtxt(baryon_pca_file)
        except:
            self.baryon_pcas = None
        self.mask        = np.loadtxt(self.config_args_emu['sampling']['scalecut_mask'])[:,1].astype(bool)
        self.mask_ones   = np.loadtxt(self.mask_ones_path)[:,1].astype(bool)
        self.dv_fid      = np.loadtxt(self.dv_fid_path)[:,1]
        self.dv_obs      = np.loadtxt(self.dv_obs_path)[:,1]
        self.output_dims = len(self.dv_obs)
        self.shear_calib_mask = np.load(self.config_args_emu['shear_calib']['mask'])
        
        # Fast parameters sequence: linear bias, shear calib, baryon PCs ?
        if self.probe != 'cosmic_shear':
            self.galaxy_bias_mask = np.load(self.config_args_emu['galaxy_bias']['mask'])
            self.n_fast_pars = self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo
        else:
            self.n_fast_pars = self.n_pcas_baryon + self.source_ntomo
        
        assert len(self.dv_obs)==len(self.dv_fid),"Observed data vector is of different size compared to the fiducial data vector."
        self.cov            = self.get_full_cov(cov_file)
        self.dv_std         = np.sqrt(np.diagonal(self.cov))
        # Add Hartlap factor to CMB lensing covariance
        self.masked_inv_cov = np.linalg.inv(self.cov[self.mask][:,self.mask])
        Nbp_remain = int(np.sum(self.mask[-self.Nbp:]))
        self.masked_inv_cov[-Nbp_remain:,-Nbp_remain:] /= self.Hartlap**2
        
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
    
    def get_full_cov(self, cov_file):
        print("Getting covariance...")
        full_cov = np.loadtxt(cov_file)
        cov = np.zeros((self.output_dims, self.output_dims))
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
