#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>

#include "../src/api.hpp"

static void *dl_handle = NULL;

static const char *ssu_get_lib_name() {
   /*
    *  TODO: Auto-detect appropriate CPU architecture
    */
   return "libssu_nv.so";
}


static void ssu_load(const char *fncname,
                     void **dl_ptr) {
   char *error;

   if (dl_handle==NULL) {
       dl_handle = dlopen(ssu_get_lib_name(), RTLD_LAZY);
       if (!dl_handle) {
          fputs(dlerror(), stderr);
          exit(1);
       }
   }

   *dl_ptr = dlsym(dl_handle, fncname);
   if ((error = dlerror()) != NULL)  {
       fputs(error, stderr);
       exit(1);
   }
}

/*********************************************************************/

static void (*dl_ssu_set_random_seed)(unsigned int) = NULL;
void ssu_set_random_seed(unsigned int new_seed) {
   if (dl_ssu_set_random_seed==NULL) ssu_load("ssu_set_random_seed", (void **) &dl_ssu_set_random_seed);

   (*dl_ssu_set_random_seed)(new_seed);
}

/*********************************************************************/

static ComputeStatus (*dl_one_off)(const char*, const char*, const char*, bool, double, bool, unsigned int, mat_t**) = NULL;
ComputeStatus one_off(const char* biom_filename, const char* tree_filename,
                             const char* unifrac_method, bool variance_adjust, double alpha,
                             bool bypass_tips, unsigned int n_substeps, mat_t** result) {
   if (dl_one_off==NULL) ssu_load("one_off", (void **) &dl_one_off);

   (*dl_one_off)(biom_filename, tree_filename, unifrac_method, variance_adjust, alpha, bypass_tips, n_substeps, result);
}

/*********************************************************************/

static ComputeStatus (*dl_faith_pd_one_off)(const char*, const char*, r_vec**);
ComputeStatus faith_pd_one_off(const char* biom_filename, const char* tree_filename,
                                      r_vec** result) {
   if (dl_faith_pd_one_off==NULL) ssu_load("faith_pd_one_off", (void **) &dl_faith_pd_one_off);

   (*dl_faith_pd_one_off)(biom_filename, tree_filename, result);
}

