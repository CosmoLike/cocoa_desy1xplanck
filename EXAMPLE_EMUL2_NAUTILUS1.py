import warnings
import os
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message=".*column is deprecated.*",
    module=r"sacc\.sacc"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*overflow encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Function not smooth or differentiabl*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Hartlap correction*"
)
import argparse, random
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from nautilus import Prior, Sampler
from getdist import loadMCSamples
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_PROJECT_NAUTILUS1')
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/lsst_y1/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="example_nautilus1")
parser.add_argument("--nlive",
                    dest="nlive",
                    help="Number of live points ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1000)
parser.add_argument("--maxfeval",
                    dest="maxfeval",
                    help="Minimizer: maximum number of likelihood evaluations",
                    type=int,
                    nargs='?',
                    const=1,
                    default=100000)
parser.add_argument("--neff",
                    dest="neff",
                    help="Minimum effective sample size. ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=10000)
parser.add_argument("--flive",
                    dest="flive",
                    help="Maximum fraction of the evidence contained in the live set before building the initial shells terminates",
                    type=float,
                    nargs='?',
                    const=1,
                    default=0.01)
parser.add_argument("--nnetworks",
                    dest="nnetworks",
                    help="Number of Neural Networks",
                    type=int,
                    nargs='?',
                    const=1,
                    default=4)
args, unknown = parser.parse_known_args()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
yaml_string=r"""
likelihood:
  desy1xplanck.cosmic_shear:
    use_emulator: 2
    path: ./external_modules/data/desy1xplanck
    data_file: Y3xPlanckPR4.dataset
    accuracyboost: 1.0
    integration_accuracy: 0
    lmax: 75000 # lmax computed for real space correlation function
    kmax_boltzmann: 7.5
    # 1 EE2, 2 Halofit (check below on the slow/fast decomposition)
    # Warning: Euclid Emulator has strict boundaries
    non_linear_emul: 2
    IA_model: 1 # NLA (0) or TATT (1)
    IA_code: 0  # CFASTPT (0) or FASTPT (1) (only works if not NLA)
    IA_redshift_evolution: 3
    debug: false
    use_baryon_pca: True
params:
  As_1e9:
    prior:
      min: 0.5
      max: 5
    ref:
      dist: norm
      loc: 2.1
      scale: 0.25
    proposal: 0.2
    latex: 10^9 A_\mathrm{s}
    renames: A
  ns:
    prior:
      min: 0.87
      max: 1.07
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.01
    proposal: 0.01
    latex: n_\mathrm{s}
  H0:
    prior:
      min: 55
      max: 91
    ref:
      dist: norm
      loc: 67.32
      scale: 5
    proposal: 3
    latex: H_0
  omegab:
    prior:
      min: 0.03
      max: 0.07
    ref:
      dist: norm
      loc: 0.0495
      scale: 0.004
    proposal: 0.004
    latex: \Omega_\mathrm{b}
  omegam:
    prior:
      min: 0.1
      max: 0.9
    ref:
      dist: norm
      loc: 0.316
      scale: 0.01
    proposal: 0.01
    latex: \Omega_\mathrm{m}
  w:
    value: -1.0
    latex: w_{0,\mathrm{DE}}
  w0pwa:
    value: -1.0
    latex: w_{0,\mathrm{DE}}+w_{a,\mathrm{DE}}
    drop: true
  wa:
    value: 'lambda w0pwa, w: w0pwa - w'
    latex: w_{a,\mathrm{DE}}
  mnu:
    value: 0.06
  omegabh2:
    value: 'lambda omegab, H0: omegab*(H0/100)**2'
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    value: 'lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708'
    latex: \Omega_\mathrm{c} h^2
  As:
    value: 'lambda As_1e9: 1e-9 * As_1e9'
    latex: A_\mathrm{s}
  # ----------------------------------------------------------------------------
  # DES-Y3 nuisance parameters begins
  # ----------------------------------------------------------------------------
  # baryon effects: the first two PCs
  # Q1 informative: [0, 7]
  # Q1 wide: [-6, 20]
  # Q2: [-4, 4] 
  DES_BARYON_Q1:
    prior:
      min: 0.0
      max: 7.0
    ref:
      dist: norm
      loc: 2.0
      scale: 0.5
    proposal: 0.5
    latex: Q_\mathrm{DES}^1
  DES_BARYON_Q2:
    prior:
      min: -4.0
      max: 4.0
    ref:
      dist: norm
      loc: 2.0
      scale: 0.5
    proposal: 0.5
    latex: Q_\mathrm{DES}^2
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # WL photo-z errors
  DES_DZ_S1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.018
    ref:
      dist: norm
      loc: 0.0
      scale: 0.018
    proposal: 0.009
    latex: \Delta z_\mathrm{s,DES}^1
  DES_DZ_S2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.015
    ref:
      dist: norm
      loc: 0.0
      scale: 0.015
    proposal: 0.0075
    latex: \Delta z_\mathrm{s,DES}^2
  DES_DZ_S3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.011
    ref:
      dist: norm
      loc: 0.0
      scale: 0.011
    proposal: 0.0055
    latex: \Delta z_\mathrm{s,DES}^3
  DES_DZ_S4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.017
    ref:
      dist: norm
      loc: 0.0
      scale: 0.017
    proposal: 0.0085
    latex: \Delta z_\mathrm{s,DES}^4
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # Intrinsic alignment
  DES_A1_1:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: 0.7
      scale: 0.5
    proposal: 0.75
    latex: A_\mathrm{1-IA,DES}^1
  DES_A1_2:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: -1.7
      scale: 0.5
    proposal: 0.75
    latex: A_\mathrm{1-IA,DES}^2
  DES_A1_3:
    value: 0
    latex: A_\mathrm{1-IA,DES}^3        
  DES_A1_4:
    value: 0
    latex: A_\mathrm{1-IA,DES}^4  
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  DES_A2_1:
    prior:
      min: -5.0
      max: 5.0
    ref:
      dist: norm
      loc: -1.36
      scale: 0.2
    proposal: 0.2
    latex: A_\mathrm{2-IA,DES}^1
  DES_A2_2:
    prior:
      min: -5.0
      max: 5.0
    ref:
      dist: norm
      loc: -2.5
      scale: 0.2
    proposal: 0.2
    latex: A_\mathrm{2-IA,DES}^2
  DES_A2_3:
    value: 0.0
    latex: A_\mathrm{2-IA,DES}^3
  DES_A2_4:
    value: 0.0
    latex: A_\mathrm{2-IA,DES}^4
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  DES_BTA_1:
    prior:
      min: 0.0
      max: 2.0
    ref:
      dist: norm
      loc: 1.0
      scale: 0.1
    proposal: 0.1
    latex: A_\mathrm{BTA-IA,DES}^1
  DES_BTA_2:
    value: 0
    latex: A_\mathrm{BTA-IA,DES}^2
  DES_BTA_3:
    value: 0
    latex: A_\mathrm{BTA-IA,DES}^3        
  DES_BTA_4:
    value: 0
    latex: A_\mathrm{BTA-IA,DES}^4
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # Shear calibration parameters
  DES_M1:
    prior:
      dist: norm
      loc: -0.0063
      scale: 0.0091
    ref:
      dist: norm
      loc: -0.0063
      scale: 0.0091
    proposal: 0.0045
    latex: m_\mathrm{DES}^1
  DES_M2:
    prior:
      dist: norm
      loc: -0.0198
      scale: 0.0078
    ref:
      dist: norm
      loc: -0.0198
      scale: 0.0078
    proposal: 0.004
    latex: m_\mathrm{DES}^2
  DES_M3:
    prior:
      dist: norm
      loc: -0.0241
      scale: 0.0076
    ref:
      dist: norm
      loc: -0.0241
      scale: 0.0076
    proposal: 0.004
    latex: m_\mathrm{DES}^3
  DES_M4:
    prior:
      dist: norm
      loc: -0.0369
      scale: 0.0076
    ref:
      dist: norm
      loc: -0.0369
      scale: 0.0076
    proposal: 0.004
    latex: m_\mathrm{DES}^4 
  # ----------------------------------------------------------------------------
  # DES-Y3 nuisance parameters ends
  # ----------------------------------------------------------------------------
theory:
  emulrdrag:
    path: ./cobaya/cobaya/theories/
    provides: ['rdrag']
    extra_args:
      file: ['external_modules/data/emultrf/BAO_SN_RES/emul_lcdm_rdrag_GP.joblib'] 
      extra: ['external_modules/data/emultrf/BAO_SN_RES/extra_lcdm_rdrag.npy'] 
      ord: [['omegabh2','omegach2']]
  emulbaosn:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    provides: ['comoving_radial_distance', 'angular_diameter_distance', 'Hubble']
    extra_args:
      device: "cuda"
      file:  [None, 'external_modules/data/emultrf/BAO_SN_RES/w0wa/emul_w0wa_H.pt']
      extra: [None, 'external_modules/data/emultrf/BAO_SN_RES/w0wa/extra_w0wa_H.npy']    
      ord: [None, ['omegam','H0','w','wa']]
      extrapar: [{'MLA': 'INT', 'ZMIN' : 0.0001, 'ZMAX' : 3, 'NZ' : 600},
                 {'MLA': 'ResMLP', 'offset' : 0.0, 'INTDIM' : 4, 'NLAYER' : 6,
                  'TMAT': 'external_modules/data/emultrf/BAO_SN_RES/w0wa/PCA_w0wa_H.npy',
                  'ZLIN': 'external_modules/data/emultrf/BAO_SN_RES/w0wa/z_lin_w0wa.npy'}]
  emulmps:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    extra_args:
      model_file:    "external_modules/data/emultrf/emulmps/npce_emul/emulator_npce.keras"
      metadata_file: "external_modules/data/emultrf/emulmps/npce_emul/metadata.joblib"
      nl_model_file:    "external_modules/data/emultrf/emulmps/w0wa_halofit/emulator_halofit.keras"
      nl_metadata_file: "external_modules/data/emultrf/emulmps/w0wa_halofit/metadata.joblib"
      use_syren: False
      param_order: ["As_1e9", "ns", "H0", "omegab", "omegam", 'w', 'wa']
#  fastpt:
#    path: ./external_modules/code/FAST-PT
#    extra_args:
#      accuracyboost: 10
#      kmax_boltzmann: 7.5
#      extrap_kmax: 250.0
"""
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
model = get_model(yaml_load(yaml_string))
def chi2(p):
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    if np.any(np.isinf(p)) or  np.any(np.isnan(p)):
      raise ValueError(f"At least one parameter value was infinite (CoCoa) param = {p}")
    point = dict(zip(model.parameterization.sampled_params(), p))
    res1 = model.logprior(point,make_finite=False)
    if np.isinf(res1) or  np.any(np.isnan(res1)):
      return 1.e20
    res2 = model.loglike(point,
                         make_finite=False,
                         cached=False,
                         return_derived=False)
    if np.isinf(res2) or  np.any(np.isnan(res2)):
      return 1e20
    return -2.0*(res1+res2)

def likelihood(params):
  res = chi2(params)
  if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
    return -np.inf
  else:
    return -0.5*res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from mpi4py.futures import MPIPoolExecutor

if __name__ == '__main__':
    print(f"nlive={args.nlive}, output={args.root}chains/{args.outroot}")
    # Build Nautilus Prior from Cobaya
    NautilusPrior = Prior()                                       # Nautilus Call 
    dim    = model.prior.d()                                      # Cobaya call
    bounds = model.prior.bounds(confidence=0.999999)              # Cobaya call
    names  = list(model.parameterization.sampled_params().keys()) # Cobaya Call
    for b, name in zip(bounds, names):
      NautilusPrior.add_parameter(name, dist=(b[0], b[1]))
    
    sampler = Sampler(NautilusPrior, 
                      likelihood,  
                      filepath=f"{args.root}chains/{args.outroot}_checkpoint.hdf5", 
                      n_dim=dim,
                      pool=MPIPoolExecutor(),
                      n_live=args.nlive,
                      n_networks=args.nnetworks,
                      resume=True)
    sampler.run(f_live=args.flive,
                n_eff=args.neff,
                n_like_max=args.maxfeval,
                verbose=True,
                discard_exploration=True)
    points, log_w, log_l = sampler.posterior()
    
    # Save output file ---------------------------------------------------------
    os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
    np.savetxt(f"{args.root}chains/{args.outroot}.1.txt",
               np.column_stack((np.exp(log_w), log_l, points, -2*log_l)),
               fmt="%.5e",
               header=f"nlive={args.nlive}, maxfeval={args.maxfeval}, log-Z ={sampler.log_z}\n"+' '.join(names),
               comments="# ")
    
    # Save a range files -------------------------------------------------------
    rows = [(str(n),float(l),float(h)) for n,l,h in zip(names,bounds[:,0],bounds[:,1])]
    with open(f"{args.root}chains/{args.outroot}.ranges", "w") as f: 
      f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

    # Save a paramname files ---------------------------------------------------
    param_info = model.info()['params']
    latex  = [param_info[x]['latex'] for x in names]
    names.append("chi2*")
    latex.append("\\chi^2")
    np.savetxt(f"{args.root}chains/{args.outroot}.paramnames", 
               np.column_stack((names,latex)),
               fmt="%s")

    # Save a cov matrix --------------------------------------------------------
    samples = loadMCSamples(f"{args.root}chains/{args.outroot}",
                            settings={'ignore_rows': u'0.0'})
    np.savetxt(f"{args.root}chains/{args.outroot}.covmat",
               np.array(samples.cov(), dtype='float64'),
               fmt="%.5e",
               header=' '.join(names),
               comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------