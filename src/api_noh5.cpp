/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2019-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

#ifdef UNIFRAC_NVIDIA
#include "api_nvidia.hpp"
#else
#include "api.hpp"
// NVIDIA header must be after for proper exports to be in place
# ifdef USE_UNIFRAC_NVIDIA
# include "api_nvidia.hpp"
# endif
#endif

#include "biom_inmem.hpp"
#include "tree.hpp"
#include "tsv.hpp"
#include "unifrac.hpp"
#include "skbio_alt.hpp"
#include <fstream>
#include <iomanip>
#include <thread>
#include <cstring>
#include <stdlib.h> 
#include <string.h> 
#include <string> 
#include <vector>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <lz4.h>
#include <time.h>

#define MMAP_FD_MASK 0x0fff
#define MMAP_FLAG    0x1000

// Note: Threading is now full controlled by OpenMP.
// Any threads variable is really referring to n_substeps.
// The old naming was retained to minimize code refactoring.

#define SETUP_TDBG(method) const char *tdbg_method=method; \
                          bool print_tdbg = false;\
                          if (const char* env_p = std::getenv("UNIFRAC_TIMING_INFO")) { \
                            print_tdbg = true; \
                            std::string env_s(env_p); \
                            if ((env_s=="NO") || (env_s=="N") || (env_s=="no") || (env_s=="n") || \
                               (env_s=="NEVER") || (env_s=="never")) print_tdbg = false; \
                          } \
                          time_t tgdb_t0; time(&tgdb_t0); \
                          if(print_tdbg) printf("INFO (unifrac): Starting %s\n",tdbg_method);

#define TDBG_STEP(sname) if(print_tdbg) {\
                           time_t tgdb_t1; time(&tgdb_t1); \
                           printf("INFO (unifrac): dt %4i : Completed %s.%s\n",(int)(tgdb_t1-tgdb_t0),tdbg_method,sname); \
                           tgdb_t0 = tgdb_t1; \
                         }

#define SET_METHOD(requested_method, err) Method method;                                                            \
                                          if(std::strcmp(requested_method, "unweighted") == 0)                      \
                                              method = unweighted_fp32;                                             \
                                          else if(std::strcmp(requested_method, "weighted_normalized") == 0)        \
                                              method = weighted_normalized_fp32;                                    \
                                          else if(std::strcmp(requested_method, "weighted_unnormalized") == 0)      \
                                              method = weighted_unnormalized_fp32;                                  \
                                          else if(std::strcmp(requested_method, "generalized") == 0)                \
                                              method = generalized_fp32;                                            \
                                          else if(std::strcmp(requested_method, "unweighted_fp64") == 0)            \
                                              method = unweighted;                                                  \
                                          else if(std::strcmp(requested_method, "weighted_normalized_fp64") == 0)   \
                                              method = weighted_normalized;                                         \
                                          else if(std::strcmp(requested_method, "weighted_unnormalized_fp64") == 0) \
                                              method = weighted_unnormalized;                                       \
                                          else if(std::strcmp(requested_method, "generalized_fp64") == 0)           \
                                              method = generalized;                                                 \
                                          else if(std::strcmp(requested_method, "unweighted_fp32") == 0)            \
                                              method = unweighted_fp32;                                             \
                                          else if(std::strcmp(requested_method, "weighted_normalized_fp32") == 0)   \
                                              method = weighted_normalized_fp32;                                    \
                                          else if(std::strcmp(requested_method, "weighted_unnormalized_fp32") == 0) \
                                              method = weighted_unnormalized_fp32;                                  \
                                          else if(std::strcmp(requested_method, "generalized_fp32") == 0)           \
                                              method = generalized_fp32;                                            \
                                          else {                                                               \
                                              return err;                                                      \
                                          }

#define SYNC_TREE_TABLE(tree, table) std::unordered_set<std::string> to_keep(table.get_obs_ids().begin(),           \
                                                                             table.get_obs_ids().end());            \
                                     su::BPTree tree_sheared = tree.shear(to_keep).collapse();

using namespace su;
using namespace std;

#ifdef UNIFRAC_NVIDIA
void ssu_set_random_seed_nv(unsigned int new_seed) {
#else
void ssu_set_random_seed(unsigned int new_seed) {
  // set it also in all the dependent sub-systems
# ifdef USE_UNIFRAC_NVIDIA
  ssu_set_random_seed_nv(new_seed);
#endif

#endif
  su::set_random_seed(new_seed);
}


void destroy_stripes(vector<double*> &dm_stripes, vector<double*> &dm_stripes_total, unsigned int n_samples,
                     unsigned int stripe_start, unsigned int stripe_stop) {
    unsigned int n_rotations = (n_samples + 1) / 2;

    if(stripe_stop == 0) {
        for(unsigned int i = 0; i < n_rotations; i++) {
            free(dm_stripes[i]);
            if(dm_stripes_total[i] != NULL)
                free(dm_stripes_total[i]);
        }
    } else {
        // if a stripe_stop is specified, and if we're in the stripe window, do not free
        // dm_stripes. this is done as the pointers in dm_stripes are assigned to the partial_mat_t
        // and subsequently freed in destroy_partial_mat. but, we do need to free dm_stripes_total
        // if appropriate
        for(unsigned int i = stripe_start; i < stripe_stop; i++) {
            if(dm_stripes_total[i] != NULL)
                free(dm_stripes_total[i]);
        }
    }
}


void initialize_mat(mat_t* &result, biom_interface &table, bool is_upper_triangle) {
    result = (mat_t*)malloc(sizeof(mat));
    result->n_samples = table.n_samples;

    result->cf_size = su::comb_2(table.n_samples);
    result->is_upper_triangle = is_upper_triangle;
    result->sample_ids = (char**)malloc(sizeof(char*) * result->n_samples);
    result->condensed_form = (double*)malloc(sizeof(double) * su::comb_2(table.n_samples));

    const std::vector<std::string> &table_sample_ids = table.get_sample_ids();
    for(unsigned int i = 0; i < result->n_samples; i++) {
        size_t len = table_sample_ids[i].length();
        result->sample_ids[i] = (char*)malloc(sizeof(char) * len + 1);
        table_sample_ids[i].copy(result->sample_ids[i], len);
        result->sample_ids[i][len] = '\0';
    }
}

void initialize_results_vec(r_vec* &result, biom_interface &table){
    // Stores results for Faith PD
    result = (r_vec*)malloc(sizeof(results_vec));
    result->n_samples = table.n_samples;
    result->values = (double*)malloc(sizeof(double) * result->n_samples);
    result->sample_ids = (char**)malloc(sizeof(char*) * result->n_samples);

    const std::vector<std::string> &table_sample_ids = table.get_sample_ids();
    for(unsigned int i = 0; i < result->n_samples; i++) {
        size_t len = table_sample_ids[i].length();
        result->sample_ids[i] = (char*)malloc(sizeof(char) * len + 1);
        table_sample_ids[i].copy(result->sample_ids[i], len);
        result->sample_ids[i][len] = '\0';
        result->values[i] = 0;
    }

}

void initialize_mat_no_biom(mat_t* &result, char** sample_ids, unsigned int n_samples, bool is_upper_triangle) {
    result = (mat_t*)malloc(sizeof(mat));
    result->n_samples = n_samples;

    result->cf_size = su::comb_2(n_samples);
    result->is_upper_triangle = is_upper_triangle;
    result->sample_ids = (char**)malloc(sizeof(char*) * result->n_samples);
    result->condensed_form = (double*)malloc(sizeof(double) * su::comb_2(n_samples));

    for(unsigned int i = 0; i < n_samples; i++) {
        result->sample_ids[i] = strdup(sample_ids[i]);
    }
}

inline compute_status is_fp64_method(const std::string &method_string, bool &fp64) {
    if ((method_string=="unweighted") || (method_string=="weighted_normalized") || (method_string=="weighted_unnormalized") || (method_string=="generalized")) {
        fp64 = false;
    } else if ((method_string=="unweighted_fp64") || (method_string=="weighted_normalized_fp64") || (method_string=="weighted_unnormalized_fp64") || (method_string=="generalized_fp64")) {
       fp64 = true;
    } else if ((method_string=="unweighted_fp32") || (method_string=="weighted_normalized_fp32") || (method_string=="weighted_unnormalized_fp32") || (method_string=="generalized_fp32")) {
       fp64 = false;
    } else {
        return unknown_method;
    }

    return okay;
}


template<class TReal, class TMat>
void initialize_mat_full_no_biom_T(TMat* &result, const char* const * sample_ids, unsigned int n_samples, 
                                   const char *mmap_dir /* if NULL or "", use malloc */) {
    result = (TMat*)malloc(sizeof(mat));
    result->n_samples = n_samples;

    uint64_t n_samples_64 = result->n_samples; // force 64bit to avoit overflow problems

    result->sample_ids = (char**)malloc(sizeof(char*) * n_samples_64);
    result->flags=0;

    if (mmap_dir!=NULL) {
     if (mmap_dir[0]==0) mmap_dir = NULL; // easier to have a simple test going on
    }

    uint64_t msize = sizeof(TReal) * n_samples_64 * n_samples_64;
    if (mmap_dir==NULL) {
      result->matrix = (TReal*)malloc(msize);
    } else {
      std::string mmap_template(mmap_dir);
      mmap_template+="/su_mmap_XXXXXX";
      // note: mkstemp/mkostemp will update mmap_template in place
#ifdef O_NOATIME
      int fd=mkostemp((char *) mmap_template.c_str(), O_NOATIME ); 
#else
      int fd=mkstemp((char *) mmap_template.c_str() );
#endif
      if (fd<0) {
         result->matrix = NULL;
         // leave error handling to the caller
      } else {
        // remove the file name, so it will be destroyed on close
        unlink(mmap_template.c_str());
        // make it big enough
        ftruncate(fd,msize);
        // now can be used, just like a malloc-ed buffer
        result->matrix = (TReal*)mmap(NULL, msize,PROT_READ|PROT_WRITE, MAP_SHARED|MAP_NORESERVE, fd, 0);
        result->flags=(uint32_t(fd) & MMAP_FD_MASK) | MMAP_FLAG;
      }
   }

    for(unsigned int i = 0; i < n_samples; i++) {
        result->sample_ids[i] = strdup(sample_ids[i]);
    }
}

void initialize_partial_mat(partial_mat_t* &result, biom_interface &table, std::vector<double*> &dm_stripes,
                            unsigned int stripe_start, unsigned int stripe_stop, bool is_upper_triangle) {
    result = (partial_mat_t*)malloc(sizeof(partial_mat));
    result->n_samples = table.n_samples;

    result->sample_ids = (char**)malloc(sizeof(char*) * result->n_samples);
    const std::vector<std::string> &table_sample_ids = table.get_sample_ids();
    for(unsigned int i = 0; i < result->n_samples; i++) {
        size_t len = table_sample_ids[i].length();
        result->sample_ids[i] = (char*)malloc(sizeof(char) * len + 1);
        table_sample_ids[i].copy(result->sample_ids[i], len);
        result->sample_ids[i][len] = '\0';
    }

    result->stripes = (double**)malloc(sizeof(double*) * (stripe_stop - stripe_start));
    result->stripe_start = stripe_start;
    result->stripe_stop = stripe_stop;
    result->is_upper_triangle = is_upper_triangle;
    result->stripe_total = dm_stripes.size();

    for(unsigned int i = stripe_start; i < stripe_stop; i++) {
        result->stripes[i - stripe_start] = dm_stripes[i];
    }
}

void destroy_results_vec(r_vec** result) {
    // for Faith PD
    for(unsigned int i = 0; i < (*result)->n_samples; i++) {
        free((*result)->sample_ids[i]);
    };
    free((*result)->sample_ids);
    free((*result)->values);
    free(*result);
}

void destroy_mat(mat_t** result) {
    for(unsigned int i = 0; i < (*result)->n_samples; i++) {
        free((*result)->sample_ids[i]);
    };
    free((*result)->sample_ids);
    if (((*result)->condensed_form)!=NULL) {
      free((*result)->condensed_form);
    }
    free(*result);
}

template<class TMat, class TReal>
inline void destroy_mat_full_T(TMat** result) {
    for(uint32_t i = 0; i < (*result)->n_samples; i++) {
        free((*result)->sample_ids[i]);   
    };                                        
    free((*result)->sample_ids);          
    if (((*result)->matrix)!=NULL) {          
      if (((*result)->flags & MMAP_FLAG) == 0)  {
         free((*result)->matrix);            
      } else {
         uint64_t n_samples = (*result)->n_samples;
         munmap((*result)->matrix, sizeof(TReal)*n_samples*n_samples);

         int fd = (*result)->flags & MMAP_FD_MASK;
         close(fd);
      }
      (*result)->matrix=NULL;
    }                                         
    free(*result);                        
}


void destroy_mat_full_fp64(mat_full_fp64_t** result) {
    destroy_mat_full_T<mat_full_fp64_t,double>(result);
}

void destroy_mat_full_fp32(mat_full_fp32_t** result) {
    destroy_mat_full_T<mat_full_fp32_t,float>(result);
}

void destroy_partial_mat(partial_mat_t** result) {
    for(unsigned int i = 0; i < (*result)->n_samples; i++) {
        if((*result)->sample_ids[i] != NULL)
            free((*result)->sample_ids[i]);
    };
    if((*result)->sample_ids != NULL)
        free((*result)->sample_ids);

    unsigned int n_stripes = (*result)->stripe_stop - (*result)->stripe_start;
    for(unsigned int i = 0; i < n_stripes; i++)
        if((*result)->stripes[i] != NULL)
            free((*result)->stripes[i]);
    if((*result)->stripes != NULL)
        free((*result)->stripes);

    free(*result);
}

void destroy_partial_dyn_mat(partial_dyn_mat_t** result) {
    for(unsigned int i = 0; i < (*result)->n_samples; i++) {
        if((*result)->sample_ids[i] != NULL)
            free((*result)->sample_ids[i]);
    };
    if((*result)->sample_ids != NULL)
        free((*result)->sample_ids);

    unsigned int n_stripes = (*result)->stripe_stop - (*result)->stripe_start;
    for(unsigned int i = 0; i < n_stripes; i++)
        if((*result)->stripes[i] != NULL)
            free((*result)->stripes[i]);
    if((*result)->stripes != NULL)
        free((*result)->stripes);
    if((*result)->offsets != NULL)
        free((*result)->offsets);
    if((*result)->filename != NULL)
        free((*result)->filename);

    free(*result);
}


void set_tasks(std::vector<su::task_parameters> &tasks,
               double alpha,
               unsigned int n_samples,
               unsigned int stripe_start,
               unsigned int stripe_stop,
               bool bypass_tips,
               unsigned int n_tasks) {

    // compute from start to the max possible stripe if stop doesn't make sense
    if(stripe_stop <= stripe_start)
        stripe_stop = (n_samples + 1) / 2;

    /* chunking strategy is to balance as much as possible. eg if there are 15 stripes
     * and 4 threads, our goal is to assign 4 stripes to 3 threads, and 3 stripes to one thread.
     *
     * we use the remaining the chunksize for bins which cannot be full maximally
     */
    unsigned int fullchunk = ((stripe_stop - stripe_start) + n_tasks - 1) / n_tasks;  // this computes the ceiling
    unsigned int smallchunk = (stripe_stop - stripe_start) / n_tasks;

    unsigned int n_fullbins = (stripe_stop - stripe_start) % n_tasks;
    if(n_fullbins == 0)
        n_fullbins = n_tasks;

    unsigned int start = stripe_start;

    for(unsigned int tid = 0; tid < n_tasks; tid++) {
        tasks[tid].tid = tid;
        tasks[tid].start = start; // stripe start
        tasks[tid].bypass_tips = bypass_tips;

        if(tid < n_fullbins) {
            tasks[tid].stop = start + fullchunk;  // stripe end
            start = start + fullchunk;
        } else {
            tasks[tid].stop = start + smallchunk;  // stripe end
            start = start + smallchunk;
        }

        tasks[tid].n_samples = n_samples;
        tasks[tid].g_unifrac_alpha = alpha;
    }
}

#ifdef UNIFRAC_NVIDIA

#define SUCMP_NM  su_acc
#include "unifrac_cmp.hpp"
#undef SUCMP_NM

// test only once, then use persistent value
static int proc_use_acc = -1;

inline bool use_acc() {
 if (proc_use_acc!=-1) return (proc_use_acc!=0);
 int has_nvidia_gpu_rc = access("/proc/driver/nvidia/gpus", F_OK);

 bool print_info = false;

 if (const char* env_p = std::getenv("UNIFRAC_GPU_INFO")) {
   print_info = true;
   std::string env_s(env_p);
   if ((env_s=="NO") || (env_s=="N") || (env_s=="no") || (env_s=="n") ||
       (env_s=="NEVER") || (env_s=="never")) {
     print_info = false;
   }
 }

 if (has_nvidia_gpu_rc == 0) {
    if (!su_acc::found_gpu()) {
       has_nvidia_gpu_rc  = 1;
       if (print_info) printf("INFO (unifrac): NVIDIA GPU listed but OpenACC cannot use it.\n");
    }
 } 

 if (has_nvidia_gpu_rc != 0) {
   if (print_info) printf("INFO (unifrac): GPU not found, using CPU\n");
   proc_use_acc=0;
   return false;
 }

 if (const char* env_p = std::getenv("UNIFRAC_USE_GPU")) {
   std::string env_s(env_p);
   if ((env_s=="NO") || (env_s=="N") || (env_s=="no") || (env_s=="n") ||
       (env_s=="NEVER") || (env_s=="never")) {
     if (print_info) printf("INFO (unifrac): Use of GPU explicitly disabled, using CPU\n");
     proc_use_acc=0;
     return false;
   }
 }

 if (print_info) printf("INFO (unifrac): Using GPU\n");
 proc_use_acc=1;
 return true;
}

bool ssu_should_use_nv() {
  return use_acc();
}

#endif // UNIFRAC_NVIDIA

compute_status one_off_inmem_cpp(su_c_biom_inmem_t &c_table_data, su_c_bptree_t &c_tree_data,
                                 const char* unifrac_method, bool variance_adjust, double alpha,
                                 bool bypass_tips, unsigned int nthreads, mat_t** result) {
    SETUP_TDBG("one_off_inmem")
    su::biom_inmem table(c_table_data);
    su::BPTree tree(c_tree_data);
    SYNC_TREE_TABLE(tree, table)
    TDBG_STEP("sync_tree_table")
    SET_METHOD(unifrac_method, unknown_method)

    const unsigned int stripe_stop = (table.n_samples + 1) / 2;
    std::vector<double*> dm_stripes(stripe_stop);
    std::vector<double*> dm_stripes_total(stripe_stop);

    if(nthreads > dm_stripes.size()) {
        fprintf(stderr, "More threads were requested than stripes. Using %d threads.\n", dm_stripes.size());
        nthreads = dm_stripes.size();
    }

    std::vector<su::task_parameters> tasks(nthreads);
    std::vector<std::thread> threads(nthreads);

    set_tasks(tasks, alpha, table.n_samples, 0, stripe_stop, bypass_tips, nthreads);
    su::process_stripes(table, tree_sheared, method, variance_adjust, dm_stripes, dm_stripes_total, threads, tasks);

    TDBG_STEP("process_stripes")
    initialize_mat(*result, table, true);  // true -> is_upper_triangle
    for(unsigned int tid = 0; tid < threads.size(); tid++) {
        su::stripes_to_condensed_form(dm_stripes,table.n_samples,(*result)->condensed_form,tasks[tid].start,tasks[tid].stop);
    }

    TDBG_STEP("stripes_to_condensed_form")
    destroy_stripes(dm_stripes, dm_stripes_total, table.n_samples, 0, 0);

    return okay;
}

compute_status partial_inmem_cpp(su_c_biom_inmem_t &c_table_data, su_c_bptree_t &c_tree_data,
                                 const char* unifrac_method, bool variance_adjust, double alpha, bool bypass_tips,
                                 unsigned int nthreads, unsigned int stripe_start, unsigned int stripe_stop,
                                 partial_mat_t** result) {

    SETUP_TDBG("partial_inmem")
    su::biom_inmem table(c_table_data);
    su::BPTree tree(c_tree_data);
    SYNC_TREE_TABLE(tree, table)
    SET_METHOD(unifrac_method, unknown_method)

    // we resize to the largest number of possible stripes even if only computing
    // partial, however we do not allocate arrays for non-computed stripes so
    // there is a little memory waste here but should be on the order of
    // 8 bytes * N samples per vector.
    std::vector<double*> dm_stripes((table.n_samples + 1) / 2);
    std::vector<double*> dm_stripes_total((table.n_samples + 1) / 2);

    if(nthreads > dm_stripes.size()) {
        fprintf(stderr, "More threads were requested than stripes. Using %d threads.\n", dm_stripes.size());
        nthreads = dm_stripes.size();
    }

    std::vector<su::task_parameters> tasks(nthreads);
    std::vector<std::thread> threads(nthreads);

    if(((table.n_samples + 1) / 2) < stripe_stop) {
        fprintf(stderr, "Stopping stripe is out-of-bounds, max %d\n", (table.n_samples + 1) / 2);
        exit(EXIT_FAILURE);
    }

    set_tasks(tasks, alpha, table.n_samples, stripe_start, stripe_stop, bypass_tips, nthreads);
    su::process_stripes(table, tree_sheared, method, variance_adjust, dm_stripes, dm_stripes_total, threads, tasks);

    TDBG_STEP("process_stripes")
    initialize_partial_mat(*result, table, dm_stripes, stripe_start, stripe_stop, true);  // true -> is_upper_triangle
    TDBG_STEP("partial_mat")
    destroy_stripes(dm_stripes, dm_stripes_total, table.n_samples, stripe_start, stripe_stop);

    return okay;
}

#ifdef UNIFRAC_NVIDIA
compute_status one_off_inmem_nv_fp64(su_c_biom_inmem_t *biom_data, su_c_bptree_t *tree_data,
                                     const char* unifrac_method, bool variance_adjust, double alpha,
                                     bool bypass_tips, unsigned int nthreads, mat_t** result) {
    SETUP_TDBG("one_off_nv")

    // condensed form
    return one_off_inmem_cpp(c_table_data, c_tree_data, unifrac_method, variance_adjust, alpha, bypass_tips, nthreads, result);
}

compute_status partial_inmem_nv(su_c_biom_inmem_t *biom_data, su_c_bptree_t *tree_data,
                                const char* unifrac_method, bool variance_adjust, double alpha, bool bypass_tips,
                                unsigned int nthreads, unsigned int stripe_start, unsigned int stripe_stop,
                                partial_mat_t** result) {
    SETUP_TDBG("partial_nv")

    return partial_inmem_cpp(c_table_data, c_tree_data,
                             unifrac_method, variance_adjust, alpha,
                             bypass_tips, nthreads, stripe_start, stripe_stop,
                             result);
}
#endif

// TMat mat_full_fp32_t
template<class TReal, class TMat>
compute_status one_off_matrix_T(su::biom_interface &table, su::BPTree &tree,
                                const char* unifrac_method, bool variance_adjust, double alpha,
                                bool bypass_tips, unsigned int nthreads,
                                const char *mmap_dir,  
                                TMat** result) {
    SETUP_TDBG("one_off_matrix_inmem")
    if (mmap_dir!=NULL) {
     if (mmap_dir[0]==0) mmap_dir = NULL; // easier to have a simple test going on
    }

    SET_METHOD(unifrac_method, unknown_method)
    SYNC_TREE_TABLE(tree, table)

    TDBG_STEP("sync_tree_table")
    const unsigned int stripe_stop = (table.n_samples + 1) / 2;
    partial_mat_t *partial_mat = NULL;

    {
      std::vector<double*> dm_stripes(stripe_stop);
      std::vector<double*> dm_stripes_total(stripe_stop);

      std::vector<su::task_parameters> tasks(nthreads);
      std::vector<std::thread> threads(nthreads);

      set_tasks(tasks, alpha, table.n_samples, 0, stripe_stop, bypass_tips, nthreads);
      su::process_stripes(table, tree_sheared, method, variance_adjust, dm_stripes, dm_stripes_total, threads, tasks);

      TDBG_STEP("process_stripes")
      initialize_partial_mat(partial_mat, table, dm_stripes, 0, stripe_stop, true);  // true -> is_upper_triangle
      if ((partial_mat==NULL) || (partial_mat->stripes==NULL) || (partial_mat->sample_ids==NULL) ) {
          fprintf(stderr, "Memory allocation error! (initialize_partial_mat)\n");
          exit(EXIT_FAILURE);
      }
      destroy_stripes(dm_stripes, dm_stripes_total, table.n_samples, 0, stripe_stop);
    }

    // allow the caller to allocate the memory
    if((*result) == NULL) {
        initialize_mat_full_no_biom_T<TReal,TMat>(*result, partial_mat->sample_ids, partial_mat->n_samples,mmap_dir);
    }

    if (((*result)==NULL) || ((*result)->matrix==NULL) || ((*result)->sample_ids==NULL) ) {
        fprintf(stderr, "Memory allocation error! (initialize_mat)\n");
        exit(EXIT_FAILURE);
    }


    {
      MemoryStripes ps(partial_mat->stripes);
      const uint32_t tile_size = (mmap_dir==NULL) ? \
                                  (128/sizeof(TReal)) : /* keep it small for memory access, to fit in chip cache */ \
                                  (4096/sizeof(TReal)); /* make it larger for mmap, as the limiting factor is swapping */
      su::stripes_to_matrix_T<TReal>(ps, partial_mat->n_samples, partial_mat->stripe_total, (*result)->matrix, tile_size);
    }
    TDBG_STEP("stripes_to_matrix")
    destroy_partial_mat(&partial_mat);

    return okay;
}

template<class TReal, class TMat>
compute_status one_off_matrix_T(su::biom_interface &table, su::BPTree &tree,
                                const char* unifrac_method, bool variance_adjust, double alpha,
                                bool bypass_tips, unsigned int nthreads,
                                unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                TMat** result) {
    if (subsample_depth>0) {
        SETUP_TDBG("one_off_matrix_subsampled")
        // We do not implement subsampling without replacement yet
        if (!subsample_with_replacement) {
          fprintf(stderr, "ERROR: subsampling without replacement not implemented yet.\n");
          return table_empty;
        }
        su::skbio_biom_subsampled table_subsampled(table, subsample_depth);
        if ((table_subsampled.n_samples==0) || (table_subsampled.n_obs==0)) {
           return table_empty;
        }
        TDBG_STEP("subsample")
        return one_off_matrix_T<TReal,TMat>(table_subsampled,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,mmap_dir,result);
    } else {
        return one_off_matrix_T<TReal,TMat>(table,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,mmap_dir,result);
    }
}

#ifdef UNIFRAC_NVIDIA
compute_status one_off_matrix_sparse_nv_fp64_v2(su_c_biom_sparse_t *table_data, su_c_bptree_t *tree_data,
                                                const char* unifrac_method, bool variance_adjust, double alpha,
                                                bool bypass_tips, unsigned int nthreads,
                                                unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                mat_full_fp64_t** result) {
    su_c_biom_sparse_t &c_table_data = *table_data;
    su_c_bptree_t &c_tree_data = *tree_data;
    SETUP_TDBG("one_off_matrix_inmem_nv")

#else
compute_status one_off_matrix_inmem_v2(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                       const char* unifrac_method, bool variance_adjust, double alpha,
                                       bool bypass_tips, unsigned int nthreads,
                                       unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                       mat_full_fp64_t** result) {
    su_c_biom_sparse_t c_table_data;
    c_table_data.obs_ids = table_data->obs_ids;
    c_table_data.sample_ids = table_data->sample_ids;
    c_table_data.indices = table_data->indices;
    c_table_data.indptr = table_data->indptr;
    c_table_data.data = table_data->data;
    c_table_data.n_obs = table_data->n_obs;
    c_table_data.n_samples = table_data->n_samples;

    su_c_bptree_t c_tree_data;
    c_tree_data.structure = tree_data->structure;
    c_tree_data.lengths =   tree_data->lengths;
    c_tree_data.names   =   tree_data->names;
    c_tree_data.n_parens =  tree_data->n_parens;
# ifdef USE_UNIFRAC_NVIDIA
   if (ssu_should_use_nv()) return one_off_matrix_sparse_nv_fp64_v2(&c_table_data, &c_tree_data,
                                              unifrac_method, variance_adjust, alpha,
                                              bypass_tips, nthreads, subsample_depth, subsample_with_replacement, mmap_dir,
                                              result);
# endif

    SETUP_TDBG("one_off_matrix_inmem")

#endif
    bool fp64;
    compute_status rc = is_fp64_method(unifrac_method, fp64);

    if (rc == okay) {
        if (!fp64) {
            return invalid_method;
        }
    } else {
        return rc;
    }

    su::biom_inmem table(c_table_data);
    su::BPTree tree(c_tree_data);
    return one_off_matrix_T<double,mat_full_fp64_t>(table, tree, unifrac_method, variance_adjust, alpha,
                                                    bypass_tips,nthreads,mmap_dir,
                                                    result);
}

#ifdef UNIFRAC_NVIDIA
compute_status one_off_matrix_sparse_nv_fp32_v2(su_c_biom_sparse_t *table_data, su_c_bptree_t *tree_data,
                                                const char* unifrac_method, bool variance_adjust, double alpha,
                                                bool bypass_tips, unsigned int nthreads,
                                                unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                mat_full_fp32_t** result) {
    su_c_biom_sparse_t &c_table_data = *table_data;
    su_c_bptree_t &c_tree_data = *tree_data;
    SETUP_TDBG("one_off_matrix_sparse_nv_fp32")

#else
compute_status one_off_matrix_inmem_fp32_v2(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                            const char* unifrac_method, bool variance_adjust, double alpha,
                                            bool bypass_tips, unsigned int nthreads,
                                            unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                            mat_full_fp32_t** result) {
    su_c_biom_sparse_t c_table_data;
    c_table_data.obs_ids = table_data->obs_ids;
    c_table_data.sample_ids = table_data->sample_ids;
    c_table_data.indices = table_data->indices;
    c_table_data.indptr = table_data->indptr;
    c_table_data.data = table_data->data;
    c_table_data.n_obs = table_data->n_obs;
    c_table_data.n_samples = table_data->n_samples;

    su_c_bptree_t c_tree_data;
    c_tree_data.structure = tree_data->structure;
    c_tree_data.lengths =   tree_data->lengths;
    c_tree_data.names   =   tree_data->names;
    c_tree_data.n_parens =  tree_data->n_parens;

# ifdef USE_UNIFRAC_NVIDIA
   if (ssu_should_use_nv()) return one_off_matrix_sparse_nv_fp32_v2(&c_table_data, &c_tree_data,
                                              unifrac_method, variance_adjust, alpha,
                                              bypass_tips, nthreads, subsample_depth, subsample_with_replacement, mmap_dir,
                                              result);
# endif

    SETUP_TDBG("one_off_matrix_inmem_fp32")

#endif
    bool fp64;
    compute_status rc = is_fp64_method(unifrac_method, fp64);

    if (rc == okay) {
        if (fp64) {
            return invalid_method;
        }
    } else {
        return rc;
    }

    su::biom_inmem table(c_table_data);
    su::BPTree tree(c_tree_data);
    return one_off_matrix_T<float,mat_full_fp32_t>(table, tree, unifrac_method, variance_adjust, alpha,
                                                   bypass_tips, nthreads, mmap_dir,
                                                   result);
}

#ifdef UNIFRAC_NVIDIA
// Don't define these functions to avoid unresolved references
#else
// Old interface
compute_status one_off_inmem(const support_biom_t *table_data, const support_bptree_t *tree_data,
                             const char* unifrac_method, bool variance_adjust, double alpha,
                             bool bypass_tips, unsigned int nthreads, mat_full_fp64_t** result) {
    return one_off_matrix_inmem_v2(table_data, tree_data, unifrac_method, variance_adjust, alpha, bypass_tips, nthreads,
                                   0, true,  NULL,
                                   result);
}

compute_status one_off_inmem_fp32(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                  const char* unifrac_method, bool variance_adjust, double alpha,
                                  bool bypass_tips, unsigned int nthreads, mat_full_fp32_t** result) {
    return one_off_matrix_inmem_fp32_v2(table_data, tree_data, unifrac_method, variance_adjust, alpha, bypass_tips, nthreads,
                                        0, true,  NULL,
                                        result);
}
#endif

#ifdef UNIFRAC_NVIDIA
compute_status one_off_matrix_inmem_nv_fp64_v2(su_c_biom_inmem_t *table_data, su_c_bptree_t *tree_data,
                                               const char* unifrac_method, bool variance_adjust, double alpha,
                                               bool bypass_tips, unsigned int nthreads,
                                               unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                               mat_full_fp64_t** result) {
    su_c_inmem_sparse_t &c_table_data = *table_data;
    su_c_bptree_t &c_tree_data = *tree_data;
    SETUP_TDBG("one_off_matrix_inmem_nv")

#else
compute_status one_off_matrix_inmem_cpp(su_c_biom_inmem_t &c_table_data, su_c_bptree_t &c_tree_data,
                                        const char* unifrac_method, bool variance_adjust, double alpha,
                                        bool bypass_tips, unsigned int nthreads,
                                        unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                        mat_full_fp64_t** result) {
# ifdef USE_UNIFRAC_NVIDIA
   if (ssu_should_use_nv()) return one_off_matrix_inmem_nv_fp64_v2(&c_table_data, &c_tree_data,
                                              unifrac_method, variance_adjust, alpha,
                                              bypass_tips, nthreads, subsample_depth, subsample_with_replacement, mmap_dir,
                                              result);
# endif

    SETUP_TDBG("one_off_matrix_inmem_cpp")

#endif
    bool fp64;
    compute_status rc = is_fp64_method(unifrac_method, fp64);

    if (rc == okay) {
        if (!fp64) {
            return invalid_method;
        }
    } else {
        return rc;
    }

    su::biom_inmem table(c_table_data);
    su::BPTree tree(c_tree_data);
    return one_off_matrix_T<double,mat_full_fp64_t>(table, tree, unifrac_method, variance_adjust, alpha,
                                                    bypass_tips,nthreads,mmap_dir,
                                                    result);
}

#ifdef UNIFRAC_NVIDIA
compute_status one_off_matrix_inmem_nv_fp32_v2(su_c_biom_inmem_t *table_data, su_c_bptree_t *tree_data,
                                               const char* unifrac_method, bool variance_adjust, double alpha,
                                               bool bypass_tips, unsigned int nthreads,
                                               unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                               mat_full_fp32_t** result) {
    su_c_biom_inmem_t &c_table_data = *table_data;
    su_c_bptree_t &c_tree_data = *tree_data;
    SETUP_TDBG("one_off_matrix_inmem_nv_fp32")

#else
compute_status one_off_matrix_inmem_cpp(su_c_biom_inmem_t &c_table_data, su_c_bptree_t &c_tree_data,
                                        const char* unifrac_method, bool variance_adjust, double alpha,
                                        bool bypass_tips, unsigned int nthreads,
                                        unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                        mat_full_fp32_t** result) {
# ifdef USE_UNIFRAC_NVIDIA
   if (ssu_should_use_nv()) return one_off_matrix_inmem_nv_fp32_v2(&c_table_data, &c_tree_data,
                                              unifrac_method, variance_adjust, alpha,
                                              bypass_tips, nthreads, subsample_depth, subsample_with_replacement, mmap_dir,
                                              result);
# endif

    SETUP_TDBG("one_off_matrix_inmem_cpp_fp32")

#endif
    bool fp64;
    compute_status rc = is_fp64_method(unifrac_method, fp64);

    if (rc == okay) {
        if (fp64) {
            return invalid_method;
        }
    } else {
        return rc;
    }

    su::biom_inmem table(c_table_data);
    su::BPTree tree(c_tree_data);
    return one_off_matrix_T<float,mat_full_fp32_t>(table, tree, unifrac_method, variance_adjust, alpha,
                                                   bypass_tips, nthreads, mmap_dir,
                                                   result);
}

