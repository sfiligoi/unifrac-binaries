#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include "../src/api.hpp"

static void *dl_handle = NULL;

static const char *ssu_get_lib_name() {
   /*
    *  TODO: Auto-detect appropriate CPU architecture
    */

   __builtin_cpu_init ();
   bool has_avx  = __builtin_cpu_supports ("avx");
   bool has_avx2 = __builtin_cpu_supports ("avx2");

   const char* env_max_cpu = getenv("UNIFRAC_MAX_CPU");

   if ((env_max_cpu!=NULL) && (strcmp(env_max_cpu,"basic")==0)) {
      has_avx = false;
      has_avx2 = false;
   }

   const char *ssu = "libssu_nv.so";
   if (has_avx) {
      if ((env_max_cpu!=NULL) && (strcmp(env_max_cpu,"avx")==0)) {
         has_avx2 = false;
      }
      if (has_avx2) {
         ssu="libssu_nv_avx2.so";
      } else {
         ssu="libssu_nv.so";
      }
   } else { // no avx
      const char* env_gpu_info = getenv("UNIFRAC_GPU_INFO");
      if ((env_gpu_info!=NULL) && (env_gpu_info[0]=='Y')) {
         printf("INFO (unifrac): CPU too old, disabling GPU\n");
      }
      ssu="libssu_cpu_basic.so";
   }

   const char* env_cpu_info = getenv("UNIFRAC_CPU_INFO");
   if ((env_cpu_info!=NULL) && (env_cpu_info[0]=='Y')) {
      printf("INFO (unifrac): Using shared library %s\n",ssu);
   }
   return ssu;
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

