#include <assert.h>
#include <gsl/gsl_interp2d.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>

#include "basics.h"
#include "baryons_JX.h"
#include "structs.h"

#include "log.c/src/log.h"
#include <hdf5.h>

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
  log_info("test_baryon: Done!");
}