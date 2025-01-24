/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

#include "../src/api.hpp"

/*
 * Implement wrappers around all the EXTERN functions
 * defined in api.hpp.
 *
 */

static pthread_mutex_t dl_mutex = PTHREAD_MUTEX_INITIALIZER;

/*********************************************************************/

/* Pick the right libssu implementation */
#ifndef BASIC_ONLY
static const char *ssu_get_lib_name() {
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

#else
static const char *ssu_get_lib_name() {
   const char *ssu = "libssu_cpu_basic.so";

   const char* env_gpu_info = getenv("UNIFRAC_GPU_INFO");
   if ((env_gpu_info!=NULL) && (env_gpu_info[0]=='Y')) {
         printf("INFO (unifrac): No GPU support in this version\n");
   }

   const char* env_cpu_info = getenv("UNIFRAC_CPU_INFO");
   if ((env_cpu_info!=NULL) && (env_cpu_info[0]=='Y')) {
      printf("INFO (unifrac): Using shared library %s\n",ssu);
   }
   return ssu;
}


#endif

/*********************************************************************/

/* Handle pointing to the approriate libssu implementing the functionality
 * Initialized on first use. */
static void *dl_handle = NULL;

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

static void cond_ssu_load(const char *fncname,
                     void **dl_ptr) {

   pthread_mutex_lock(&dl_mutex);
   if ((*dl_ptr)==NULL) ssu_load(fncname,dl_ptr);
   pthread_mutex_unlock(&dl_mutex);
}

/*********************************************************************/
/* All the functons below are wrappers
 * and each has its own function pointer
 * that is initialized on first use */
/*********************************************************************/

static void (*dl_ssu_set_random_seed)(unsigned int) = NULL;
void ssu_set_random_seed(unsigned int new_seed) {
   cond_ssu_load("ssu_set_random_seed", (void **) &dl_ssu_set_random_seed);

   (*dl_ssu_set_random_seed)(new_seed);
}

/*********************************************************************/

static void (*dl_destroy_mat)(mat_t**) = NULL;
static void (*dl_destroy_mat_full_fp64)(mat_full_fp64_t**) = NULL;
static void (*dl_destroy_mat_full_fp32)(mat_full_fp32_t**) = NULL;
static void (*dl_destroy_partial_mat)(partial_mat_t**) = NULL;
static void (*dl_destroy_partial_dyn_mat)(partial_dyn_mat_t**) = NULL;
static void (*dl_destroy_results_vec)(r_vec**) = NULL;
static void (*dl_destroy_bptree_opaque)(opaque_bptree_t**) = NULL;

static IOStatus (*dl_read_bptree_opaque)(const char*, opaque_bptree_t**) = NULL;
static void (*dl_load_bptree_opaque)(const char*, opaque_bptree_t**) = NULL;
static void (*dl_convert_bptree_opaque)(const support_bptree_t*, opaque_bptree_t**) = NULL;
static int (*dl_get_bptree_opaque_els)(opaque_bptree_t*) = NULL;

void destroy_mat(mat_t** result) {
   cond_ssu_load("destroy_mat", (void **) &dl_destroy_mat);

   (*dl_destroy_mat)(result);
}

void destroy_mat_full_fp64(mat_full_fp64_t** result) {
   cond_ssu_load("destroy_mat_full_fp64", (void **) &dl_destroy_mat_full_fp64);

   (*dl_destroy_mat_full_fp64)(result);
}

void destroy_mat_full_fp32(mat_full_fp32_t** result) {
   cond_ssu_load("destroy_mat_full_fp32", (void **) &dl_destroy_mat_full_fp32);

   (*dl_destroy_mat_full_fp32)(result);
}

void destroy_partial_mat(partial_mat_t** result) {
   cond_ssu_load("destroy_partial_mat", (void **) &dl_destroy_partial_mat);

   (*dl_destroy_partial_mat)(result);
}

void destroy_partial_dyn_mat(partial_dyn_mat_t** result) {
   cond_ssu_load("destroy_partial_dyn_mat", (void **) &dl_destroy_partial_dyn_mat);

   (*dl_destroy_partial_dyn_mat)(result);
}

void destroy_results_vec(r_vec** result) {
   cond_ssu_load("destroy_results_vec", (void **) &dl_destroy_results_vec);

   (*dl_destroy_results_vec)(result);
}

void destroy_bptree_opaque(opaque_bptree_t** tree_data) {
   cond_ssu_load("destroy_bptree_opaque", (void **) &dl_destroy_bptree_opaque);

   (*dl_destroy_bptree_opaque)(tree_data);
}

IOStatus read_bptree_opaque(const char* tree_filename, opaque_bptree_t** tree_data) {
   cond_ssu_load("read_bptree_opaque", (void **) &dl_read_bptree_opaque);

   return (*dl_read_bptree_opaque)(tree_filename,tree_data);
}

void load_bptree_opaque(const char* newick, opaque_bptree_t** tree_data) {
   cond_ssu_load("load_bptree_opaque", (void **) &dl_load_bptree_opaque);

   (*dl_load_bptree_opaque)(newick,tree_data);
}

void convert_bptree_opaque(const support_bptree_t* in_tree, opaque_bptree_t** tree_data) {
   cond_ssu_load("convert_bptree_opaque", (void **) &dl_convert_bptree_opaque);

   (*dl_convert_bptree_opaque)(in_tree,tree_data);
}

int get_bptree_opaque_els(opaque_bptree_t* tree_data) {
   cond_ssu_load("get_bptree_opaque_els", (void **) &dl_get_bptree_opaque_els);

   return (*dl_get_bptree_opaque_els)(tree_data);
}

/*********************************************************************/

static ComputeStatus (*dl_one_off)(const char*, const char*, const char*, bool, double, bool, unsigned int, mat_t**) = NULL;
static ComputeStatus (*dl_one_off_wtree)(const char*, const opaque_bptree_t*, const char*, bool, double, bool, unsigned int, mat_t**) = NULL;

ComputeStatus one_off(const char* biom_filename, const char* tree_filename,
                             const char* unifrac_method, bool variance_adjust, double alpha,
                             bool bypass_tips, unsigned int n_substeps, mat_t** result) {
   cond_ssu_load("one_off", (void **) &dl_one_off);

   return (*dl_one_off)(biom_filename, tree_filename, unifrac_method, variance_adjust, alpha, bypass_tips, n_substeps, result);
}

ComputeStatus one_off_wtree(const char* biom_filename, const opaque_bptree_t* tree_data,
                                   const char* unifrac_method, bool variance_adjust, double alpha,
                                   bool bypass_tips, unsigned int n_substeps, mat_t** result) {
   cond_ssu_load("one_off_wtree", (void **) &dl_one_off_wtree);

   return (*dl_one_off_wtree)(biom_filename, tree_data, unifrac_method, variance_adjust, alpha, bypass_tips, n_substeps, result);
}

/*********************************************************************/

static ComputeStatus (*dl_one_off_matrix_inmem_v2)(const support_biom_t *, const support_bptree_t *, const char*, bool, double, 
                                                   bool, unsigned int, unsigned int, bool, const char *, mat_full_fp64_t**) = NULL;
static ComputeStatus (*dl_one_off_inmem)(const support_biom_t *, const support_bptree_t *, const char*, bool, double,
                                         bool, unsigned int, mat_full_fp64_t**) = NULL;
static ComputeStatus (*dl_one_off_matrix_inmem_fp32_v2)(const support_biom_t *, const support_bptree_t *, const char*, bool, double,
                                                        bool, unsigned int, unsigned int, bool, const char *, mat_full_fp32_t**) = NULL;
static ComputeStatus (*dl_one_off_inmem_fp32)(const support_biom_t *, const support_bptree_t *, const char*, bool, double,
                                              bool, unsigned int, mat_full_fp32_t**) = NULL;

ComputeStatus one_off_matrix_inmem_v2(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                             const char* unifrac_method, bool variance_adjust, double alpha,
                                             bool bypass_tips, unsigned int n_substeps,
                                             unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                             mat_full_fp64_t** result) {
   cond_ssu_load("one_off_matrix_inmem_v2", (void **) &dl_one_off_matrix_inmem_v2);

   return (*dl_one_off_matrix_inmem_v2)(table_data, tree_data, unifrac_method, variance_adjust, alpha,
                                 bypass_tips, n_substeps, subsample_depth, subsample_with_replacement, mmap_dir, result);
}

ComputeStatus one_off_inmem(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                   const char* unifrac_method, bool variance_adjust, double alpha,
                                   bool bypass_tips, unsigned int n_substeps, mat_full_fp64_t** result) {
   cond_ssu_load("one_off_inmem", (void **) &dl_one_off_inmem);

   return (*dl_one_off_inmem)(table_data, tree_data, unifrac_method, variance_adjust, alpha, bypass_tips, n_substeps, result);
}

ComputeStatus one_off_matrix_inmem_fp32_v2(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                                  const char* unifrac_method, bool variance_adjust, double alpha,
                                                  bool bypass_tips, unsigned int n_substeps,
                                                  unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                  mat_full_fp32_t** result) {
   cond_ssu_load("one_off_matrix_inmem_fp32_v2", (void **) &dl_one_off_matrix_inmem_fp32_v2);

   return (*dl_one_off_matrix_inmem_fp32_v2)(table_data, tree_data, unifrac_method, variance_adjust, alpha,
                                      bypass_tips, n_substeps, subsample_depth, subsample_with_replacement, mmap_dir, result);
}

ComputeStatus one_off_inmem_fp32(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                        const char* unifrac_method, bool variance_adjust, double alpha,
                                        bool bypass_tips, unsigned int n_substeps, mat_full_fp32_t** result) {
   cond_ssu_load("one_off_inmem_fp32", (void **) &dl_one_off_inmem_fp32);

   return (*dl_one_off_inmem_fp32)(table_data, tree_data, unifrac_method, variance_adjust, alpha, bypass_tips, n_substeps, result);
}

/*********************************************************************/

static ComputeStatus (*dl_one_off_matrix_v2)(const char*, const char*, const char*, bool, double,
                                             bool, unsigned int, unsigned int, bool, const char *, mat_full_fp64_t**) = NULL;
static ComputeStatus (*dl_one_off_matrix_v2t)(const char*, const opaque_bptree_t*, const char*, bool, double,
                                             bool, unsigned int, unsigned int, bool, const char *, mat_full_fp64_t**) = NULL;
static ComputeStatus (*dl_one_off_matrix)(const char*, const char*, const char*, bool, double,
                                          bool, unsigned int, const char *, mat_full_fp64_t**) = NULL;
static ComputeStatus (*dl_one_off_matrix_fp32_v2)(const char*, const char*, const char*, bool, double,
                                                  bool, unsigned int, unsigned int, bool, const char *, mat_full_fp32_t**) = NULL;
static ComputeStatus (*dl_one_off_matrix_fp32_v2t)(const char*, const opaque_bptree_t*, const char*, bool, double,
                                                  bool, unsigned int, unsigned int, bool, const char *, mat_full_fp32_t**) = NULL;
static ComputeStatus (*dl_one_off_matrix_fp32)(const char*, const char*, const char*, bool, double,
                                               bool, unsigned int, const char *, mat_full_fp32_t**) = NULL;

ComputeStatus one_off_matrix_v2(const char* biom_filename, const char* tree_filename,
                                       const char* unifrac_method, bool variance_adjust, double alpha,
                                       bool bypass_tips, unsigned int n_substeps,
                                       unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                       mat_full_fp64_t** result) {
   cond_ssu_load("one_off_matrix_v2", (void **) &dl_one_off_matrix_v2);

   return (*dl_one_off_matrix_v2)(biom_filename, tree_filename, unifrac_method, variance_adjust, alpha,
                           bypass_tips, n_substeps, subsample_depth, subsample_with_replacement, mmap_dir, result);
}

ComputeStatus one_off_matrix_v2t(const char* biom_filename, const opaque_bptree_t* tree_data,
                                        const char* unifrac_method, bool variance_adjust, double alpha,
                                        bool bypass_tips, unsigned int n_substeps,
                                        unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                        mat_full_fp64_t** result) {
   cond_ssu_load("one_off_matrix_v2t", (void **) &dl_one_off_matrix_v2t);

   return (*dl_one_off_matrix_v2t)(biom_filename, tree_data, unifrac_method, variance_adjust, alpha,
                           bypass_tips, n_substeps, subsample_depth, subsample_with_replacement, mmap_dir, result);
}

ComputeStatus one_off_matrix(const char* biom_filename, const char* tree_filename,
                                    const char* unifrac_method, bool variance_adjust, double alpha,
                                    bool bypass_tips, unsigned int n_substeps,
                                    const char *mmap_dir,
                                    mat_full_fp64_t** result) {
   cond_ssu_load("one_off_matrix", (void **) &dl_one_off_matrix);

   return (*dl_one_off_matrix)(biom_filename, tree_filename, unifrac_method, variance_adjust, alpha,
                        bypass_tips, n_substeps, mmap_dir, result);
}

ComputeStatus one_off_matrix_fp32_v2(const char* biom_filename, const char* tree_filename,
                                            const char* unifrac_method, bool variance_adjust, double alpha,
                                            bool bypass_tips, unsigned int n_substeps,
                                            unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                            mat_full_fp32_t** result) {
   cond_ssu_load("one_off_matrix_fp32_v2", (void **) &dl_one_off_matrix_fp32_v2);

   return (*dl_one_off_matrix_fp32_v2)(biom_filename, tree_filename, unifrac_method, variance_adjust, alpha,
                                bypass_tips, n_substeps, subsample_depth, subsample_with_replacement, mmap_dir, result);
}

ComputeStatus one_off_matrix_fp32_v2t(const char* biom_filename, const opaque_bptree_t* tree_data,
                                             const char* unifrac_method, bool variance_adjust, double alpha,
                                             bool bypass_tips, unsigned int n_substeps,
                                             unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                             mat_full_fp32_t** result) {
   cond_ssu_load("one_off_matrix_fp32_v2t", (void **) &dl_one_off_matrix_fp32_v2t);

   return (*dl_one_off_matrix_fp32_v2t)(biom_filename, tree_data, unifrac_method, variance_adjust, alpha,
                                bypass_tips, n_substeps, subsample_depth, subsample_with_replacement, mmap_dir, result);
}

ComputeStatus one_off_matrix_fp32(const char* biom_filename, const char* tree_filename,
                                         const char* unifrac_method, bool variance_adjust, double alpha,
                                         bool bypass_tips, unsigned int n_substeps,
                                         const char *mmap_dir,
                                         mat_full_fp32_t** result) {
   cond_ssu_load("one_off_matrix_fp32", (void **) &dl_one_off_matrix_fp32);

   return (*dl_one_off_matrix_fp32)(biom_filename, tree_filename, unifrac_method, variance_adjust, alpha,
                             bypass_tips, n_substeps, mmap_dir, result);
}

/*********************************************************************/

static ComputeStatus (*dl_faith_pd_one_off)(const char*, const char*, r_vec**) = NULL;
ComputeStatus faith_pd_one_off(const char* biom_filename, const char* tree_filename,
                                      r_vec** result) {
   cond_ssu_load("faith_pd_one_off", (void **) &dl_faith_pd_one_off);

   return (*dl_faith_pd_one_off)(biom_filename, tree_filename, result);
}

/*********************************************************************/

static ComputeStatus (*dl_unifrac_to_file_v2)(const char*, const char*, const char*, const char*, bool, double,
                                              bool, unsigned int, const char*, unsigned int, bool, 
                                              unsigned int, unsigned int, const char *, const char *, const char *) = NULL;
static ComputeStatus (*dl_unifrac_to_file)(const char*, const char*, const char*, const char*, bool, double,
                                           bool, unsigned int, const char*, unsigned int, const char *) = NULL;
static ComputeStatus (*dl_unifrac_multi_to_file_v2)(const char*, const char*, const char*, const char*, bool, double,
                                              bool, unsigned int, const char*, unsigned int, unsigned int, bool, 
                                              unsigned int, unsigned int, const char *, const char *, const char *) = NULL;

ComputeStatus unifrac_to_file_v2(const char* biom_filename, const char* tree_filename, const char* out_filename,
                                        const char* unifrac_method, bool variance_adjust, double alpha,
                                        bool bypass_tips, unsigned int n_substeps, const char* format,
                                        unsigned int subsample_depth, bool subsample_with_replacement, 
                                        unsigned int pcoa_dims,
                                        unsigned int permanova_perms, const char *grouping_filename, const char *grouping_columns,
                                        const char *mmap_dir){
   cond_ssu_load("unifrac_to_file_v2", (void **) &dl_unifrac_to_file_v2);

   return (*dl_unifrac_to_file_v2)(biom_filename, tree_filename, out_filename, unifrac_method, variance_adjust, alpha,
                            bypass_tips, n_substeps, format, subsample_depth, subsample_with_replacement,
                            pcoa_dims, permanova_perms, grouping_filename, grouping_columns, mmap_dir);
}

ComputeStatus unifrac_to_file(const char* biom_filename, const char* tree_filename, const char* out_filename,
                                     const char* unifrac_method, bool variance_adjust, double alpha,
                                     bool bypass_tips, unsigned int n_substeps, const char* format,
                                     unsigned int pcoa_dims, const char *mmap_dir) {
   cond_ssu_load("unifrac_to_file", (void **) &dl_unifrac_to_file);

   return (*dl_unifrac_to_file)(biom_filename, tree_filename, out_filename, unifrac_method, variance_adjust, alpha, 
                         bypass_tips, n_substeps, format, pcoa_dims, mmap_dir);
}

ComputeStatus unifrac_multi_to_file_v2(const char* biom_filename, const char* tree_filename, const char* out_filename,
                                              const char* unifrac_method, bool variance_adjust, double alpha,
                                              bool bypass_tips, unsigned int n_substeps, const char* format,
                                              unsigned int n_subsamples, unsigned int subsample_depth, bool subsample_with_replacement, 
                                              unsigned int pcoa_dims,
                                              unsigned int permanova_perms, const char *grouping_filename, const char *grouping_columns,
                                              const char *mmap_dir) {
   cond_ssu_load("unifrac_multi_to_file_v2", (void **) &dl_unifrac_multi_to_file_v2);

   return (*dl_unifrac_multi_to_file_v2)(biom_filename, tree_filename, out_filename, unifrac_method, variance_adjust, alpha,
                                  bypass_tips, n_substeps, format, n_subsamples, subsample_depth, subsample_with_replacement,
                                  pcoa_dims, permanova_perms, grouping_filename, grouping_columns, mmap_dir);
}


/*********************************************************************/

static ComputeStatus (*dl_compute_permanova_fp64)(const char *, unsigned int, const char**, mat_full_fp64_t *, unsigned int, double *, double *) = NULL;
static ComputeStatus (*dl_compute_permanova_fp32)(const char *, unsigned int, const char**, mat_full_fp32_t *, unsigned int, float *, float *) = NULL;

ComputeStatus compute_permanova_fp64(const char *grouping_filename, unsigned int n_columns, const char* *columns,
                                            mat_full_fp64_t * result, unsigned int permanova_perms,
                                            double *fstats, double *pvalues) {
   cond_ssu_load("compute_permanova_fp64", (void **) &dl_compute_permanova_fp64);

   return (*dl_compute_permanova_fp64)(grouping_filename, n_columns, columns, result, permanova_perms, fstats, pvalues);
}

ComputeStatus compute_permanova_fp32(const char *grouping_filename, unsigned int n_columns, const char* * columns,
                                            mat_full_fp32_t * result, unsigned int permanova_perms,
                                            float *fstats, float *pvalues) {
   cond_ssu_load("compute_permanova_fp32", (void **) &dl_compute_permanova_fp32);

   return (*dl_compute_permanova_fp32)(grouping_filename, n_columns, columns, result, permanova_perms, fstats, pvalues);
}

/*********************************************************************/

static IOStatus (*dl_write_mat)(const char*, mat_t*) = NULL;
static IOStatus (*dl_write_mat_from_matrix)(const char*, mat_full_fp64_t*) = NULL;
static IOStatus (*dl_write_vec)(const char*, r_vec*) = NULL;

IOStatus write_mat(const char* filename, mat_t* result) {
   cond_ssu_load("write_mat", (void **) &dl_write_mat);

   return (*dl_write_mat)(filename, result);
}

IOStatus write_mat_from_matrix(const char* filename, mat_full_fp64_t* result) {
   cond_ssu_load("write_mat_from_matrix", (void **) &dl_write_mat_from_matrix);

   return (*dl_write_mat_from_matrix)(filename, result);
}

IOStatus write_vec(const char* filename, r_vec* result) {
   cond_ssu_load("write_vec", (void **) &dl_write_vec);

   return (*dl_write_vec)(filename, result);
}

/*********************************************************************/

static IOStatus (*dl_write_mat_from_matrix_hdf5_fp64_v2)(const char*, mat_full_fp64_t*, unsigned int, int, unsigned int,
                                                         const char* *, const char**, const double *, const double *, const unsigned int *,
                                                         const char**, const unsigned int *) = NULL;
static IOStatus (*dl_write_mat_from_matrix_hdf5_fp64)(const char*, mat_full_fp64_t*, unsigned int, int) = NULL;
static IOStatus (*dl_write_mat_from_matrix_hdf5_fp32_v2)(const char*, mat_full_fp32_t*, unsigned int, int, unsigned int,
                                                         const char**, const char**, const float *, const float *, const unsigned int *,
                                                         const char**, const unsigned int *) = NULL;
static IOStatus (*dl_write_mat_from_matrix_hdf5_fp32)(const char*, mat_full_fp32_t*, unsigned int, int) = NULL;

IOStatus write_mat_from_matrix_hdf5_fp64_v2(const char* output_filename, mat_full_fp64_t* result,
                                                   unsigned int pcoa_dims, int save_dist,
                                                   unsigned int stat_n_vals,
                                                   const char*  *stat_method_arr,     const char*        *stat_name_arr,
                                                   const double *stat_val_arr,        const double       *stat_pval_arr, const unsigned int *stat_perm_count_arr,
                                                   const char*  *stat_group_name_arr, const unsigned int *stat_group_count_arr) {
   cond_ssu_load("write_mat_from_matrix_hdf5_fp64_v2", (void **) &dl_write_mat_from_matrix_hdf5_fp64_v2);

   return (*dl_write_mat_from_matrix_hdf5_fp64_v2)(output_filename, result, pcoa_dims, save_dist, stat_n_vals,
                                            stat_method_arr, stat_name_arr, stat_val_arr, stat_pval_arr, stat_perm_count_arr,
                                            stat_group_name_arr, stat_group_count_arr);
}

IOStatus write_mat_from_matrix_hdf5_fp64(const char* filename, mat_full_fp64_t* result, unsigned int pcoa_dims, int save_dist) {
   cond_ssu_load("write_mat_from_matrix_hdf5_fp64", (void **) &dl_write_mat_from_matrix_hdf5_fp64);

   return (*dl_write_mat_from_matrix_hdf5_fp64)(filename, result, pcoa_dims, save_dist);
}

IOStatus write_mat_from_matrix_hdf5_fp32_v2(const char* output_filename, mat_full_fp32_t* result,
                                                   unsigned int pcoa_dims, int save_dist,
                                                   unsigned int stat_n_vals,
                                                   const char*  *stat_method_arr,     const char*        *stat_name_arr,
                                                   const float  *stat_val_arr,        const float        *stat_pval_arr, const unsigned int *stat_perm_count_arr,
                                                   const char*  *stat_group_name_arr, const unsigned int *stat_group_count_arr) {
   cond_ssu_load("write_mat_from_matrix_hdf5_fp32_v2", (void **) &dl_write_mat_from_matrix_hdf5_fp32_v2);

   return (*dl_write_mat_from_matrix_hdf5_fp32_v2)(output_filename, result, pcoa_dims, save_dist, stat_n_vals,
                                            stat_method_arr, stat_name_arr, stat_val_arr, stat_pval_arr, stat_perm_count_arr,
                                            stat_group_name_arr, stat_group_count_arr);
}

IOStatus write_mat_from_matrix_hdf5_fp32(const char* filename, mat_full_fp32_t* result, unsigned int pcoa_dims, int save_dist) {
   cond_ssu_load("write_mat_from_matrix_hdf5_fp32", (void **) &dl_write_mat_from_matrix_hdf5_fp32);

   return (*dl_write_mat_from_matrix_hdf5_fp32)(filename, result, pcoa_dims, save_dist);
}


/*********************************************************************/

static ComputeStatus (*dl_one_dense_pair_v2t)(unsigned int, const char **, const double*,const double*,const opaque_bptree_t*,const char*, bool, double,bool, double*) = NULL;
static ComputeStatus (*dl_one_dense_pair_v2)(unsigned int, const char **, const double*,const double*,const support_bptree_t*,const char*, bool, double,bool, double*) = NULL;

ComputeStatus one_dense_pair_v2t(unsigned int n_obs, const char ** obs_ids, const double* sample1, const double* sample2,
		                        const opaque_bptree_t* tree_data,
                                        const char* unifrac_method, bool variance_adjust, double alpha,
                                        bool bypass_tips, double* result) {
   cond_ssu_load("one_dense_pair_v2t", (void **) &dl_one_dense_pair_v2t);

   return (*dl_one_dense_pair_v2t)(n_obs,obs_ids,sample1,sample2,tree_data,unifrac_method,variance_adjust,alpha,bypass_tips,result);
}

ComputeStatus one_dense_pair_v2(unsigned int n_obs, const char ** obs_ids, const double* sample1, const double* sample2,
		                       const support_bptree_t* tree_data,
                                       const char* unifrac_method, bool variance_adjust, double alpha,
                                       bool bypass_tips, double* result) {
   cond_ssu_load("one_dense_pair_v2", (void **) &dl_one_dense_pair_v2);

   return (*dl_one_dense_pair_v2)(n_obs,obs_ids,sample1,sample2,tree_data,unifrac_method,variance_adjust,alpha,bypass_tips,result);
}

/*********************************************************************/

static ComputeStatus (*dl_partial)(const char*, const char*, const char*, bool, double, bool, unsigned int, unsigned int, unsigned int, partial_mat_t**) = NULL;
static MergeStatus (*dl_merge_partial_to_mmap_matrix)(partial_dyn_mat_t**, int, const char *, mat_full_fp64_t**) = NULL;
static MergeStatus (*dl_merge_partial_to_mmap_matrix_fp32)(partial_dyn_mat_t**, int, const char *, mat_full_fp32_t**) = NULL;
static MergeStatus (*dl_validate_partial)(const partial_dyn_mat_t* const *, int);
static IOStatus (*dl_read_partial)(const char*, partial_mat_t**);
static IOStatus (*dl_read_partial_header)(const char*, partial_dyn_mat_t**);
static IOStatus (*dl_read_partial_one_stripe)(partial_dyn_mat_t*, uint32_t);
static IOStatus (*dl_write_partial)(const char*, const partial_mat_t*);


ComputeStatus partial(const char* biom_filename, const char* tree_filename,
                             const char* unifrac_method, bool variance_adjust, double alpha,
                             bool bypass_tips, unsigned int n_substeps, unsigned int stripe_start,
                             unsigned int stripe_stop, partial_mat_t** result) {
   cond_ssu_load("partial", (void **) &dl_partial);

   return (*dl_partial)(biom_filename,tree_filename,unifrac_method,variance_adjust,alpha,
		   bypass_tips,n_substeps,stripe_start,stripe_stop,result);
}

MergeStatus merge_partial_to_mmap_matrix(partial_dyn_mat_t* * partial_mats, int n_partials, const char *mmap_dir, mat_full_fp64_t** result) {
   cond_ssu_load("merge_partial_to_mmap_matrix", (void **) &dl_merge_partial_to_mmap_matrix);

   return (*dl_merge_partial_to_mmap_matrix)(partial_mats,n_partials,mmap_dir,result);
}

MergeStatus merge_partial_to_mmap_matrix_fp32(partial_dyn_mat_t* * partial_mats, int n_partials, const char *mmap_dir, mat_full_fp32_t** result) {
   cond_ssu_load("merge_partial_to_mmap_matrix_fp32", (void **) &dl_merge_partial_to_mmap_matrix_fp32);

   return (*dl_merge_partial_to_mmap_matrix_fp32)(partial_mats,n_partials,mmap_dir,result);
}

MergeStatus validate_partial(const partial_dyn_mat_t* const * partial_mats, int n_partials) {
   cond_ssu_load("validate_partial", (void **) &dl_validate_partial);

   return (*dl_validate_partial)(partial_mats,n_partials);
}

IOStatus read_partial(const char* input_filename, partial_mat_t** result_out) {
   cond_ssu_load("read_partial", (void **) &dl_read_partial);

   return (*dl_read_partial)(input_filename,result_out);
}

IOStatus read_partial_header(const char* input_filename, partial_dyn_mat_t** result_out) {
   cond_ssu_load("read_partial_header", (void **) &dl_read_partial_header);

   return (*dl_read_partial_header)(input_filename,result_out);
}

IOStatus read_partial_one_stripe(partial_dyn_mat_t* result, uint32_t stripe_idx) {
   cond_ssu_load("read_partial_one_stripe", (void **) &dl_read_partial_one_stripe);

   return (*dl_read_partial_one_stripe)(result,stripe_idx);
}

IOStatus write_partial(const char* filename, const partial_mat_t* result) {
   cond_ssu_load("write_partial", (void **) &dl_write_partial);

   return (*dl_write_partial)(filename,result);
}


