#include <iostream>
#include <sstream>

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

void print_SPk(const char* sim_name, int sim_id){
  // Test dataset reading
  std::ostringstream oss;
  oss << "test_Baryon_read_" << sim_name << "_" << sim_id << "_HDF5C.txt";
  std::string output_fn = oss.str();
  std::ofstream output(output_fn);
  if (!output){
    std::cerr << "Error opening file." << std::endl;
    exit(1);
  }
  output << "# z lgk lgPkR" << std::endl;
  for (int i=0; i<bary.Na_bins; i++){
    for (int j=0; j<bary.Nk_bins; j++){
      output << 1.0/bary.a_bins[i]-1.0 << " " << bary.logk_bins[j] << " " << bary.log_PkR[j*bary.Na_bins+i] << std::endl;
    }
  }
  output.close();
}

// Test HDF5 reading routine
int main(){
  init_baryons("TNG100", 1);print_SPk("TNG100", 1);
  init_baryons("HzAGN", 1);print_SPk("HzAGN", 1);
  init_baryons("mb2", 1);print_SPk("mb2", 1);
  init_baryons("illustris", 1);print_SPk("illustris", 1);
  init_baryons("eagle", 1);print_SPk("eagle", 1);
  init_baryons("cowls_AGN", 1);print_SPk("cowls_AGN", 1);
  init_baryons("cowls_AGN", 2);print_SPk("cowls_AGN", 2);
  init_baryons("cowls_AGN", 3);print_SPk("cowls_AGN", 3);
  init_baryons("BAHAMAS", 1);print_SPk("BAHAMAS", 1);
  init_baryons("BAHAMAS", 2);print_SPk("BAHAMAS", 2);
  init_baryons("BAHAMAS", 3);print_SPk("BAHAMAS", 3);
  for (int i = 1; i< 401; i++){
    init_baryons("antilles", i);
    print_SPk("antilles", i);
  }
  std::cout << "test_baryon: Done!" << std::endl;
}