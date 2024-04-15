#include <iostream>

#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/baryons_JX.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/recompute.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"


// Test HDF5 reading routine
int main(){
  init_baryons("TNG100", 1);
  init_baryons("HzAGN", 1);
  init_baryons("mb2", 1);
  init_baryons("illustris", 1);
  init_baryons("eagle", 1);
  init_baryons("cowls_AGN", 1);
  init_baryons("cowls_AGN", 2);
  init_baryons("cowls_AGN", 3);
  init_baryons("BAHAMAS", 1);
  init_baryons("BAHAMAS", 2);
  init_baryons("BAHAMAS", 3);
  for (int i = 1; i< 401; i++) init_baryons("antilles", i);
  std::cout << "test_baryon: Done!" << std::endl;
}