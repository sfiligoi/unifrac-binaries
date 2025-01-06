#include "api.hpp"
#include "biom.hpp"
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

#define CHECK_FILE(filename, err) if(!is_file_exists(filename)) { \
                                      return err;                 \
                                  }

#define SET_METHOD(requested_method, err) Method method;                                                            \
                                          if(std::strcmp(requested_method, "unweighted") == 0)                      \
                                              method = unweighted_fp32;                                             \
                                          else if(std::strcmp(requested_method, "weighted_normalized") == 0)        \
                                              method = weighted_normalized_fp32;                                    \
                                          else if(std::strcmp(requested_method, "weighted_unnormalized") == 0)      \
                                              method = weighted_unnormalized_fp32;                                  \
                                          else if(std::strcmp(requested_method, "unweighted_unnormalized") == 0)    \
                                              method = unweighted_unnormalized_fp32;                                \
                                          else if(std::strcmp(requested_method, "generalized") == 0)                \
                                              method = generalized_fp32;                                            \
                                          else if(std::strcmp(requested_method, "unweighted_fp64") == 0)            \
                                              method = unweighted;                                                  \
                                          else if(std::strcmp(requested_method, "weighted_normalized_fp64") == 0)   \
                                              method = weighted_normalized;                                         \
                                          else if(std::strcmp(requested_method, "weighted_unnormalized_fp64") == 0) \
                                              method = weighted_unnormalized;                                       \
                                          else if(std::strcmp(requested_method, "unweighted_unnormalized_fp64") == 0) \
                                              method = unweighted_unnormalized;                                     \
                                          else if(std::strcmp(requested_method, "generalized_fp64") == 0)           \
                                              method = generalized;                                                 \
                                          else if(std::strcmp(requested_method, "unweighted_fp32") == 0)            \
                                              method = unweighted_fp32;                                             \
                                          else if(std::strcmp(requested_method, "weighted_normalized_fp32") == 0)   \
                                              method = weighted_normalized_fp32;                                    \
                                          else if(std::strcmp(requested_method, "weighted_unnormalized_fp32") == 0) \
                                              method = weighted_unnormalized_fp32;                                  \
                                          else if(std::strcmp(requested_method, "unweighted_unnormalized_fp32") == 0) \
                                              method = unweighted_unnormalized_fp32;                                \
                                          else if(std::strcmp(requested_method, "generalized_fp32") == 0)           \
                                              method = generalized_fp32;                                            \
                                          else {                                                               \
                                              return err;                                                      \
                                          }

#define SYNC_TREE_TABLE(tree, table) std::unordered_set<std::string> to_keep(table.get_obs_ids().begin(),           \
                                                                             table.get_obs_ids().end());            \
                                     su::BPTree tree_sheared = tree.shear(to_keep).collapse();

#define VALIDATE_TREE_TABLE(tree, table)                if(table.n_samples <= 0 | table.n_obs <= 0) {                            \
                                                            return table_empty;                                                  \
                                                        }                                                                        \
                                                        std::string bad_id = su::test_table_ids_are_subset_of_tree(table, tree); \
                                                        if(bad_id != "") {                                                       \
                                                            return table_and_tree_do_not_overlap;                                \
                                                        }   

#define PARSE_TREE_TABLE(tree_filename, table_filename) su::BPTree tree(get_tree_content(tree_filename)); \
                                                        su::biom table(biom_filename);                    \
							VALIDATE_TREE_TABLE(tree, table)

#define PARSE_SYNC_TREE_TABLE(tree_filename, table_filename) PARSE_TREE_TABLE(tree_filename, table_filename) \
                                                             SYNC_TREE_TABLE(tree, table)

using namespace su;
using namespace std;

void ssu_set_random_seed(unsigned int new_seed) {
  su::set_random_seed(new_seed);
}

// https://stackoverflow.com/a/19841704/19741
bool is_file_exists(const char *fileName) {
    std::ifstream infile(fileName);
        return infile.good();
}

inline std::string get_tree_content(const char* tree_filename) {
    std::ifstream ifs(tree_filename);
    std::string content = std::string(std::istreambuf_iterator<char>(ifs),
                                      std::istreambuf_iterator<char>());
    return content;
}

IOStatus load_bptree_opaque(const char* tree_filename, opaque_bptree_t* tree_data) {
    SETUP_TDBG("load_bptree_opaque")
    if(tree_data==NULL) return unexpected_end;
    CHECK_FILE(tree_filename, open_error)
    tree_data->opaque = (void*) new su::BPTree(get_tree_content(tree_filename));
    TDBG_STEP("load_bptree_opaque")
    return read_okay;
}

void destroy_bptree_opaque(opaque_bptree_t* tree_data) {
	if (tree_data!=NULL) {
		if (tree_data->opaque!=NULL) {
			su::BPTree *tree = (su::BPTree *) tree_data->opaque;
			tree_data->opaque = NULL;
			delete tree;
		}
	}
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
    if ((method_string=="unweighted") || (method_string=="weighted_normalized") || (method_string=="weighted_unnormalized") || (method_string=="unweighted_unnormalized") || (method_string=="generalized")) {
        fp64 = false;
    } else if ((method_string=="unweighted_fp64") || (method_string=="weighted_normalized_fp64") || (method_string=="weighted_unnormalized_fp64") || (method_string=="unweighted_unnormalized_fp64") || (method_string=="generalized_fp64")) {
       fp64 = true;
    } else if ((method_string=="unweighted_fp32") || (method_string=="weighted_normalized_fp32") || (method_string=="weighted_unnormalized_fp32") || (method_string=="unweighted_unnormalized_fp32") || (method_string=="generalized_fp32")) {
       fp64 = false;
    } else {
        return unknown_method;
    }

    return okay;
}


inline compute_status is_fp64(const std::string &method_string, const std::string &format_string, bool &fp64, bool &save_dist) {
  if (format_string == "hdf5_fp32") {
    fp64 = false;
    save_dist = true;
  } else if (format_string == "hdf5_fp64") {
    fp64 = true;
    save_dist = true;
  } else if (format_string == "hdf5") {
    save_dist = true;
    return is_fp64_method(method_string, fp64);
  } else if (format_string == "hdf5_nodist") {
    save_dist = false;
    return is_fp64_method(method_string, fp64);
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

compute_status one_off_inmem_cpp(su::biom_interface &table, const su::BPTree &tree,
                             const char* unifrac_method, bool variance_adjust, double alpha,
                             bool bypass_tips, unsigned int nthreads, mat_t** result) {
    SETUP_TDBG("one_off_inmem")
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

compute_status partial(const char* biom_filename, const char* tree_filename,
                       const char* unifrac_method, bool variance_adjust, double alpha, bool bypass_tips,
                       unsigned int nthreads, unsigned int stripe_start, unsigned int stripe_stop,
                       partial_mat_t** result) {

    SETUP_TDBG("partial")
    CHECK_FILE(biom_filename, table_missing)
    CHECK_FILE(tree_filename, tree_missing)
    SET_METHOD(unifrac_method, unknown_method)
    PARSE_SYNC_TREE_TABLE(tree_filename, table_filename)

    TDBG_STEP("load_files")
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

compute_status faith_pd_one_off(const char* biom_filename, const char* tree_filename,
                                r_vec** result){
    SETUP_TDBG("faith_pd_one_off")
    CHECK_FILE(biom_filename, table_missing)
    CHECK_FILE(tree_filename, tree_missing)
    PARSE_TREE_TABLE(tree_filename, biom_filename)
    TDBG_STEP("load_files")

    // Filter out any elements with zero counts
    su::biom_inmem table_nz(table,1.0);
    if ((table_nz.n_samples==0) || (table_nz.n_obs==0)) {
      fprintf(stderr, "WARNING: All samples had zero counts. Forcing zero result.\n");
      SYNC_TREE_TABLE(tree, table)

      TDBG_STEP("sync_tree_table")
      initialize_results_vec(*result, table);
      // nothing else to do... results already initialized to 0
      TDBG_STEP("faith_pd")
    } else {
      if ((table_nz.n_samples!=table.n_samples) || (table_nz.n_obs!=table.n_obs)) {
        fprintf(stderr, "WARNING: Some samples had zero counts and were filtered out.\n");
      }
      SYNC_TREE_TABLE(tree, table_nz)

      TDBG_STEP("sync_tree_table")
    
      initialize_results_vec(*result, table_nz);

      // compute faithpd
      su::faith_pd(table_nz, tree_sheared, std::ref((*result)->values));
      TDBG_STEP("faith_pd")
    }

    return okay;
}

compute_status one_off(const char* biom_filename, const char* tree_filename,
                       const char* unifrac_method, bool variance_adjust, double alpha,
                       bool bypass_tips, unsigned int nthreads, mat_t** result) {
    SETUP_TDBG("one_off")
    CHECK_FILE(biom_filename, table_missing)
    CHECK_FILE(tree_filename, tree_missing)
    PARSE_TREE_TABLE(tree_filename, table_filename)

    TDBG_STEP("load_files")
    // condensed form
    return one_off_inmem_cpp(table, tree, unifrac_method, variance_adjust, alpha, bypass_tips, nthreads, result);
}

compute_status one_off_wtree(const char* biom_filename, const opaque_bptree_t* tree_data,
                             const char* unifrac_method, bool variance_adjust, double alpha,
                             bool bypass_tips, unsigned int n_substeps, mat_t** result) {
    SETUP_TDBG("one_off_wtree")
    if (tree_data==NULL) return tree_missing;
    if (tree_data->opaque==NULL) return tree_missing;
    CHECK_FILE(biom_filename, table_missing)
    const su::BPTree &tree = *( (su::BPTree*) tree_data->opaque);
    su::biom table(biom_filename);
    VALIDATE_TREE_TABLE(tree, table)

    TDBG_STEP("load_files")
    // condensed form
    return one_off_inmem_cpp(table, tree, unifrac_method, variance_adjust, alpha, bypass_tips, n_substeps, result);
}

// TMat mat_full_fp32_t
template<class TReal, class TMat>
compute_status one_off_matrix_T(su::biom_interface &table, const su::BPTree &tree,
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
compute_status one_off_matrix_v2_T(su::biom_inmem &table, const su::BPTree &tree,
                                   const char* unifrac_method, bool variance_adjust, double alpha,
                                   bool bypass_tips, unsigned int nthreads,
                                   unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                   TMat** result) {
    SETUP_TDBG("one_off_matrix_inmem_v2")
    if (subsample_depth>0) {
        su::skbio_biom_subsampled table_subsampled(table, subsample_with_replacement, subsample_depth);
        if ((table_subsampled.n_samples==0) || (table_subsampled.n_obs==0)) {
           return table_empty;
        }
        TDBG_STEP("subsample")
        return one_off_matrix_T<TReal,TMat>(table_subsampled,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,mmap_dir,result);
    } else {
        return one_off_matrix_T<TReal,TMat>(table,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,mmap_dir,result);
    }
}

compute_status one_off_matrix_v2(const char* biom_filename, const char* tree_filename,
                                 const char* unifrac_method, bool variance_adjust, double alpha,
                                 bool bypass_tips, unsigned int nthreads,
                                 unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                 mat_full_fp64_t** result) {
    SETUP_TDBG("one_off_matrix")
    CHECK_FILE(biom_filename, table_missing)
    CHECK_FILE(tree_filename, tree_missing)
    PARSE_TREE_TABLE(tree_filename, biom_filename)
    TDBG_STEP("load_files")
    return one_off_matrix_v2_T<double,mat_full_fp64_t>(table,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,subsample_depth,subsample_with_replacement,mmap_dir,result);
}

compute_status one_off_matrix_fp32_v2(const char* biom_filename, const char* tree_filename,
                                      const char* unifrac_method, bool variance_adjust, double alpha,
                                      bool bypass_tips, unsigned int nthreads,
                                      unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                      mat_full_fp32_t** result) {
    SETUP_TDBG("one_off_matrix_fp32")
    CHECK_FILE(biom_filename, table_missing)
    CHECK_FILE(tree_filename, tree_missing)
    PARSE_TREE_TABLE(tree_filename, biom_filename)
    TDBG_STEP("load_files")
    return one_off_matrix_v2_T<float,mat_full_fp32_t>(table,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,subsample_depth,subsample_with_replacement,mmap_dir,result);
}

// Old interface
compute_status one_off_matrix(const char* biom_filename, const char* tree_filename,
                              const char* unifrac_method, bool variance_adjust, double alpha,
                              bool bypass_tips, unsigned int nthreads,
                              const char *mmap_dir,
                              mat_full_fp64_t** result) {
    return one_off_matrix_v2(biom_filename,tree_filename,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,0,true,mmap_dir,result);
}

// Old interface
compute_status one_off_matrix_fp32(const char* biom_filename, const char* tree_filename,
                                   const char* unifrac_method, bool variance_adjust, double alpha,
                                   bool bypass_tips, unsigned int nthreads,
                                   const char *mmap_dir,
                                   mat_full_fp32_t** result) {
    return one_off_matrix_fp32_v2(biom_filename,tree_filename,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,0,true,mmap_dir,result);
}

compute_status one_off_matrix_inmem_v2(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                       const char* unifrac_method, bool variance_adjust, double alpha,
                                       bool bypass_tips, unsigned int nthreads,
                                       unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                       mat_full_fp64_t** result) {
    SETUP_TDBG("one_off_matrix_inmem")
    bool fp64;
    compute_status rc = is_fp64_method(unifrac_method, fp64);

    if (rc == okay) {
        if (!fp64) {
            return invalid_method;
        }
    } else {
        return rc;
    }

    if(table_data->n_samples <= 0 | table_data->n_obs <= 0) {
	return table_empty;
    }

    su::biom_inmem table(table_data->obs_ids, 
                         table_data->sample_ids, 
                         table_data->indices,
                         table_data->indptr,
                         table_data->data,
                         table_data->n_obs,
                         table_data->n_samples);

    su::BPTree tree(tree_data->structure,
                    tree_data->lengths,
                    tree_data->names,
                    tree_data->n_parens);

    VALIDATE_TREE_TABLE(tree,table)

    return one_off_matrix_v2_T<double,mat_full_fp64_t>(table,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,subsample_depth,subsample_with_replacement,mmap_dir,result);
}

// Old interface
compute_status one_off_inmem(const support_biom_t *table_data, const support_bptree_t *tree_data,
                             const char* unifrac_method, bool variance_adjust, double alpha,
                             bool bypass_tips, unsigned int nthreads, mat_full_fp64_t** result) {
    return one_off_matrix_inmem_v2(table_data, tree_data, unifrac_method, variance_adjust, alpha, bypass_tips, nthreads,
                                   0, true,  NULL,
                                   result);
}

compute_status one_off_matrix_inmem_fp32_v2(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                            const char* unifrac_method, bool variance_adjust, double alpha,
                                            bool bypass_tips, unsigned int nthreads,
                                            unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                            mat_full_fp32_t** result) {
    SETUP_TDBG("one_off_matrix_inmem_fp32")
    bool fp64;
    compute_status rc = is_fp64_method(unifrac_method, fp64);

    if (rc == okay) {
        if (fp64) {
            return invalid_method;
        }
    } else {
        return rc;
    }

    if(table_data->n_samples <= 0 | table_data->n_obs <= 0) {
	return table_empty;
    }

    su::biom_inmem table(table_data->obs_ids, 
                         table_data->sample_ids, 
                         table_data->indices,
                         table_data->indptr,
                         table_data->data,
                         table_data->n_obs,
                         table_data->n_samples);

    su::BPTree tree(tree_data->structure,
                    tree_data->lengths,
                    tree_data->names,
                    tree_data->n_parens);

    VALIDATE_TREE_TABLE(tree,table)

    return one_off_matrix_v2_T<float,mat_full_fp32_t>(table,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nthreads,subsample_depth,subsample_with_replacement,mmap_dir,result);
}

// Old interface
compute_status one_off_inmem_fp32(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                  const char* unifrac_method, bool variance_adjust, double alpha,
                                  bool bypass_tips, unsigned int nthreads, mat_full_fp32_t** result) {
    return one_off_matrix_inmem_fp32_v2(table_data, tree_data, unifrac_method, variance_adjust, alpha, bypass_tips, nthreads,
                                        0, true,  NULL,
                                        result);
}

// Internal 
inline std::vector<std::string> stringlist_to_vector(const char *stringlist) {
  char *str = strdup(stringlist);
  unsigned int n_els = 1;
  for (const char *p = str; p[0]!=0; ++p) {
    if (p[0]==',') n_els++;
  }

  std::vector<std::string> out(n_els);
  char *prev=str;
  n_els = 0;
  for (char *p = str; p[0]!=0; ++p) {
    if (p[0]==',') {
       p[0] = 0;
       out[n_els] = prev;
       n_els++;
       prev = p+1;
    }
  }
  out[n_els] = prev;

  free(str);
  return out;  
}


// Internal: Make sure TReal and real_id match
template<class TReal, class TMat>
inline compute_status compute_permanova_T(const char *grouping_filename, unsigned int n_columns, const char* const* columns,
                                          TMat * result, unsigned int permanova_perms,
                                          TReal *fstats, TReal *pvalues, uint32_t *n_groups) {
     const uint32_t n_samples = result->n_samples;
     uint32_t *grouping = new uint32_t[n_samples];

     indexed_tsv tsv_obj(grouping_filename, n_samples, result->sample_ids);

     // compute each column separately
     for (unsigned int i=0; i<n_columns; i++) {
       try {
         tsv_obj.read_grouping(columns[i], grouping, n_groups[i]);
       } catch(...) {
         delete[] grouping;
         return grouping_missing;
       }

       // always use double for intermediate compute, adds trivial cost
       double my_fstat;
       double my_pvalue;
       su::permanova(result->matrix, n_samples,
                     grouping, permanova_perms,
                     my_fstat,my_pvalue);
       fstats[i] = my_fstat;
       pvalues[i] = my_pvalue;
     }
     delete[] grouping;

     return okay;
}

compute_status compute_permanova_fp64(const char *grouping_filename, unsigned int n_columns, const char* *columns,
                                      mat_full_fp64_t * result, unsigned int permanova_perms,
                                      double *fstats, double *pvalues, unsigned int *n_groups) {
  return compute_permanova_T<double,mat_full_fp64_t>(grouping_filename,n_columns,columns,result,permanova_perms,fstats,pvalues,n_groups);
}

compute_status compute_permanova_fp32(const char *grouping_filename, unsigned int n_columns, const char* *columns,
                                      mat_full_fp32_t * result, unsigned int permanova_perms,
                                      float *fstats, float *pvalues, unsigned int *n_groups) {
  return compute_permanova_T<float,mat_full_fp32_t>(grouping_filename,n_columns,columns,result,permanova_perms,fstats,pvalues,n_groups);
}

compute_status unifrac_to_file_v2(const char* biom_filename, const char* tree_filename, const char* out_filename,
                                  const char* unifrac_method, bool variance_adjust, double alpha,
                                  bool bypass_tips, unsigned int threads, const char* format,
                                  unsigned int subsample_depth, bool subsample_with_replacement,
                                  unsigned int pcoa_dims,
                                  unsigned int permanova_perms, const char *grouping_filename, const char *grouping_columns,
                                  const char *mmap_dir)
{
    SETUP_TDBG("unifrac_to_file")

    bool fp64;
    bool save_dist;
    compute_status rc = is_fp64(unifrac_method, format, fp64, save_dist);

    if (rc==okay) {
      if (fp64) {
        mat_full_fp64_t* result = NULL;
        rc = one_off_matrix_v2(biom_filename, tree_filename,
                               unifrac_method, variance_adjust, alpha,
                               bypass_tips, threads, subsample_depth, subsample_with_replacement, mmap_dir,
                               &result);
        TDBG_STEP("matrix_fp64 computed")

        if (rc==okay) {
          if (permanova_perms>0) {
            typedef const char* Tcstring;

            const auto columns = stringlist_to_vector(grouping_columns);
            const unsigned int n_columns = columns.size();
            Tcstring *columns_c = new Tcstring[n_columns];
            for (unsigned int i=0; i<n_columns; i++)  columns_c[i] = columns[i].c_str();
            double *fstats = new double[n_columns];
            double *pvalues = new double[n_columns];
            uint32_t *n_groups = new uint32_t[n_columns];

            rc = compute_permanova_fp64(grouping_filename,n_columns,columns_c,result,permanova_perms,fstats,pvalues,n_groups);
            TDBG_STEP("permanova_fp64 computed")

            if (rc==okay) {
              std::string stat_method("PERMANOVA");
              Tcstring *stat_methods = new Tcstring[n_columns];
              for (unsigned int i=0; i<n_columns; i++)  stat_methods[i] = stat_method.c_str();
              std::string stat_name("pseudo-F");
              Tcstring *stat_names = new Tcstring[n_columns];
              for (unsigned int i=0; i<n_columns; i++)  stat_names[i] = stat_name.c_str();
              uint32_t *nperm_arr = new uint32_t[n_columns];
              for (unsigned int i=0; i<n_columns; i++)  nperm_arr[i] = permanova_perms;

              IOStatus iostatus = write_mat_from_matrix_hdf5_fp64_v2(out_filename, result, pcoa_dims, save_dist,
                                                                     n_columns, stat_methods, stat_names,
                                                                     fstats, pvalues, nperm_arr,
                                                                     columns_c, n_groups);
              TDBG_STEP("file saved")
              if (iostatus!=write_okay) rc=output_error;
              delete[] nperm_arr;
              delete[] stat_methods;
              delete[] stat_names;
            }
            delete[] n_groups;
            delete[] pvalues;
            delete[] fstats;
            delete[] columns_c;
          } else {
            IOStatus iostatus = write_mat_from_matrix_hdf5_fp64(out_filename, result, pcoa_dims, save_dist);
            TDBG_STEP("file saved")
            if (iostatus!=write_okay) rc=output_error;
          }
          destroy_mat_full_fp64(&result);
        }
      } else {
        mat_full_fp32_t* result = NULL;
        rc = one_off_matrix_fp32_v2(biom_filename, tree_filename,
                                    unifrac_method, variance_adjust, alpha,
                                    bypass_tips, threads, subsample_depth, subsample_with_replacement, mmap_dir,
                                    &result);
        TDBG_STEP("matrix_fp32 computed")
     
        if (rc==okay) {
          if (permanova_perms>0) {
            typedef const char* Tcstring;

            const auto columns = stringlist_to_vector(grouping_columns);
            const unsigned int n_columns = columns.size();
            Tcstring *columns_c = new Tcstring[n_columns];
            for (unsigned int i=0; i<n_columns; i++)  columns_c[i] = columns[i].c_str();
            float *fstats = new float[n_columns];
            float *pvalues = new float[n_columns];
            uint32_t *n_groups = new uint32_t[n_columns];

            rc = compute_permanova_fp32(grouping_filename,n_columns,columns_c,result,permanova_perms,fstats,pvalues,n_groups);
            TDBG_STEP("permanova_fp32 computed")

            if (rc==okay) {
              std::string stat_method("PERMANOVA");
              Tcstring *stat_methods = new Tcstring[n_columns];
              for (unsigned int i=0; i<n_columns; i++)  stat_methods[i] = stat_method.c_str();
              std::string stat_name("pseudo-F");
              Tcstring *stat_names = new Tcstring[n_columns];
              for (unsigned int i=0; i<n_columns; i++)  stat_names[i] = stat_name.c_str();
              uint32_t *nperm_arr = new uint32_t[n_columns];
              for (unsigned int i=0; i<n_columns; i++)  nperm_arr[i] = permanova_perms;

              IOStatus iostatus = write_mat_from_matrix_hdf5_fp32_v2(out_filename, result, pcoa_dims, save_dist,
                                                                     n_columns, stat_methods, stat_names,
                                                                     fstats, pvalues, nperm_arr,
                                                                     columns_c, n_groups);
              TDBG_STEP("file saved")
              if (iostatus!=write_okay) rc=output_error;

              delete[] nperm_arr;
              delete[] stat_methods;
              delete[] stat_names;
            }
            delete[] n_groups;
            delete[] pvalues;
            delete[] fstats;
            delete[] columns_c;
          } else {
            IOStatus iostatus = write_mat_from_matrix_hdf5_fp32(out_filename, result, pcoa_dims, save_dist);
            TDBG_STEP("file saved")
            if (iostatus!=write_okay) rc=output_error;
          }
          destroy_mat_full_fp32(&result);
        }
      }
    }

    return rc;
}

// for backwards compatibility
compute_status unifrac_to_file(const char* biom_filename, const char* tree_filename, const char* out_filename,
                               const char* unifrac_method, bool variance_adjust, double alpha,
                               bool bypass_tips, unsigned int threads, const char* format,
                               unsigned int pcoa_dims, const char *mmap_dir) {
  return unifrac_to_file_v2(biom_filename,tree_filename,out_filename,unifrac_method,variance_adjust,alpha,bypass_tips,
                            threads,format,0,true,pcoa_dims,0,NULL,NULL,mmap_dir);
}

herr_t write_hdf5_string(hid_t output_file_id,const char *dname, const char *str)
{
  // this is the convoluted way to store a string
  // Will use the FORTRAN forma, so we do not depend on null termination
  hid_t filetype_id = H5Tcopy (H5T_FORTRAN_S1);
  H5Tset_size(filetype_id, strlen(str));
  hid_t memtype_id = H5Tcopy (H5T_C_S1);
  H5Tset_size(memtype_id, strlen(str)+1);

  hsize_t  dims[1] = {1};
  hid_t dataspace_id = H5Screate_simple (1, dims, NULL);

  hid_t dataset_id = H5Dcreate(output_file_id,dname, filetype_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                                H5P_DEFAULT);
  herr_t status = H5Dwrite(dataset_id, memtype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, str);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Tclose(memtype_id);
  H5Tclose(filetype_id);

  return status;
}

// Internal
inline herr_t write_hdf5_stringarray(hid_t output_file_id,
                                     const char *label,
                                     hsize_t n_els, const char * const *els) {
  hsize_t   dims[1];
  dims[0] = n_els;
  hid_t dataspace_id = H5Screate_simple(1, dims, NULL);

  // this is the convoluted way to store an array of strings
  hid_t datatype_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(datatype_id,H5T_VARIABLE);

  hid_t dcpl_id = H5Pcreate (H5P_DATASET_CREATE);

  hid_t dataset_id = H5Dcreate1(output_file_id, label , datatype_id, dataspace_id, dcpl_id);

  herr_t status = H5Dwrite(dataset_id, datatype_id, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, els);

  H5Dclose(dataset_id);
  H5Tclose(datatype_id);
  H5Sclose(dataspace_id);
  H5Pclose(dcpl_id);

  return status;
}

// Internal: Make sure TReal and real_id match
template<class TReal>
inline herr_t write_hdf5_array(hid_t output_file_id, hid_t real_id,
                               const char *label,
                               hsize_t n_els, const TReal *els) {
  hsize_t   dims[1];
  dims[0] = n_els;
  hid_t dataspace_id = H5Screate_simple(1, dims, NULL);

  hid_t dcpl_id = H5Pcreate (H5P_DATASET_CREATE);

  hid_t dataset_id = H5Dcreate2(output_file_id, label, real_id, dataspace_id,
                                H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
  herr_t status = H5Dwrite(dataset_id, real_id, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           els);

  H5Pclose(dcpl_id);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  return status;
}

// Internal: Make sure TReal and real_id match
template<class TReal>
inline herr_t write_hdf5_array2D(hid_t output_file_id, hid_t real_id,
                                 const char *label,
                                 hsize_t dim1, hsize_t dim2, const TReal *els) {
  hsize_t   dims[2];
  dims[0] = dim1;
  dims[1] = dim2;
  hid_t dataspace_id = H5Screate_simple(2, dims, NULL);

  hid_t dcpl_id = H5Pcreate (H5P_DATASET_CREATE);

  hid_t dataset_id = H5Dcreate2(output_file_id, label ,real_id, dataspace_id,
                                H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
  herr_t status = H5Dwrite(dataset_id, real_id, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           els);

  H5Pclose(dcpl_id);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  return status;
}

template<class TReal>
inline IOStatus append_hdf5_pcoa(hid_t output_file_id, hid_t real_id, unsigned int pcoa_dims, unsigned int n_samples,
                                 const char *eigenvalues_key, const char * samples_key, const char * proportion_explained_key,
                                 const TReal * eigenvalues, const TReal * samples,  const TReal * proportion_explained) {
     // save the eigenvalues
     {
       herr_t status = write_hdf5_array<TReal>(output_file_id,real_id,
                                               eigenvalues_key, pcoa_dims, eigenvalues);
       if (status<0)  return write_error;
     }

     // save the proportion_explained
     {
       herr_t status = write_hdf5_array<TReal>(output_file_id,real_id,
                                               proportion_explained_key, pcoa_dims, proportion_explained);
       if (status<0)  return write_error;
     }

     // save the samples
     {
       herr_t status = write_hdf5_array2D<TReal>(output_file_id,real_id,
                                                 samples_key, n_samples, pcoa_dims, samples);
       if (status<0)  return write_error;
     }

  return write_okay;
}

namespace su {

template<class TReal, class TMat>
class WriteHDF5Multi {
protected:
   hid_t real_id;
   std::string fname;
   unsigned int pcoa_dims;
   hid_t output_file_id;
   unsigned int n_results;
public:
   WriteHDF5Multi(hid_t _real_id, const char* output_filename, unsigned int _pcoa_dims) 
   : real_id(_real_id)
   , fname(output_filename)
   , pcoa_dims(_pcoa_dims)
   , output_file_id(H5Fcreate(output_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT))
   , n_results(0)
   {
      if (output_file_id<0) throw "File open failed";

      // simple header
      if (write_hdf5_string(output_file_id,"format","SUJK")<0) {
         H5Fclose (output_file_id);
         throw "Header write failed";
      }
      if (write_hdf5_string(output_file_id,"version","2023.03")<0) {
         H5Fclose (output_file_id);
         throw "Header write failed";
      }
      if (pcoa_dims>0) {
        if (write_hdf5_string(output_file_id,"pcoa_method","FSVD")<0) {
           H5Fclose (output_file_id);
           throw "Header write failed";
        }
      }
   }

   virtual ~WriteHDF5Multi()
   {
      H5Fclose (output_file_id);
   }

   // Note: May destroy the content of result->matrix
   void add_result(TMat * result, bool save_dist) {
      SETUP_TDBG("WriteHDF5Multi_add_result")

      char fmtstr[64];
      const auto n_samples = result->n_samples;

      if (n_results==0) {
         // save the ids once, they do not change
         {
           herr_t status = write_hdf5_stringarray(output_file_id,
                               "order", n_samples, result->sample_ids);
           if (status<0) throw "Order write failed";
         }
      }

      if (save_dist) {
         // save the matrix
         {
           sprintf(fmtstr,"matrix:%d",n_results);
           herr_t status = write_hdf5_array2D<TReal>(output_file_id,real_id,
                            fmtstr, n_samples, n_samples, result->matrix);
           if (status<0) throw "Matrix write failed";
         }
         TDBG_STEP("matrix saved")
      }

      if (pcoa_dims>0) {
         // compute pcoa and save it in the file
         // use inplace variant to keep memory use in check; we don't need matrix anymore
         TReal * eigenvalues;
         TReal * samples;
         TReal * proportion_explained;

         su::pcoa_inplace(result->matrix, n_samples, pcoa_dims, eigenvalues, samples, proportion_explained);
         TDBG_STEP("pcoa computed")

         char fmtstr2[64];
         char fmtstr3[64];
         sprintf(fmtstr,"pcoa_eigvals:%d",n_results);
         sprintf(fmtstr2,"pcoa_samples:%d",n_results);
         sprintf(fmtstr3,"pcoa_proportion_explained:%d",n_results);
         IOStatus rc = append_hdf5_pcoa(output_file_id, real_id, pcoa_dims, n_samples,
                             fmtstr, fmtstr2, fmtstr3,
                             eigenvalues, samples,  proportion_explained);
         free(eigenvalues);
         free(proportion_explained);
         free(samples);
         if (rc!=write_okay) throw "PCOA write failed";
         TDBG_STEP("pcoa saved")
      } // if pcoa

      n_results++;
   }

   void write_stats(unsigned int           stat_n_vals,
                    const char* const    * stat_method_arr, const char* const  * stat_name_arr,
                    const TReal          * stat_val_arr,    const TReal        * stat_pval_arr, const uint32_t  * stat_perm_count_arr,
                    const char* const    * stat_group_name_arr, const uint32_t * stat_group_count_arr) {
     SETUP_TDBG("WriteHDF5Multi_write_stats")
     herr_t status = write_hdf5_stringarray(output_file_id,
                         "stat_methods", stat_n_vals, stat_method_arr);
     if (status>=0) {
       status = write_hdf5_stringarray(output_file_id,
                         "stat_test_names", stat_n_vals, stat_name_arr);
     }
     if (status>=0) {
       status = write_hdf5_stringarray(output_file_id,
                         "stat_grouping_names", stat_n_vals, stat_group_name_arr);
     }
     if (status>=0) {
       status = write_hdf5_array<uint32_t>(output_file_id, H5T_STD_U32LE,
                         "stat_n_groups", stat_n_vals, stat_group_count_arr);
     }
     if (status>=0) {
       status = write_hdf5_array<TReal>(output_file_id,real_id,
                         "stat_values", stat_n_vals, stat_val_arr);
     }
     if (status>=0) {
       status = write_hdf5_array<TReal>(output_file_id,real_id,
                         "stat_pvalues", stat_n_vals, stat_pval_arr);
     }
     if (status>=0) {
       status = write_hdf5_array<uint32_t>(output_file_id, H5T_STD_U32LE,
                         "stat_n_permutations", stat_n_vals, stat_perm_count_arr);
     }
     
     // check status after cleanup, for simplicity
     if (status<0) throw "Stats write failed";
     TDBG_STEP("stats saved")
   }

};

class WriteHDF5MultiFP64 : public WriteHDF5Multi<double,mat_full_fp64_t> {
public:
   WriteHDF5MultiFP64(const char* output_filename, unsigned int _pcoa_dims)
   : WriteHDF5Multi(H5T_IEEE_F64LE, output_filename, _pcoa_dims) {}
};

class WriteHDF5MultiFP32 : public WriteHDF5Multi<float,mat_full_fp32_t> {
public:
   WriteHDF5MultiFP32(const char* output_filename, unsigned int _pcoa_dims)
   : WriteHDF5Multi(H5T_IEEE_F32LE, output_filename, _pcoa_dims) {}
};

} // end namespace

template<class TReal, class TMat>
compute_status unifrac_multi_to_file_T(hid_t real_id, const bool save_dist,
                                        const char* biom_filename, const char* tree_filename, const char* out_filename,
                                        const char* unifrac_method, bool variance_adjust, double alpha,
                                        bool bypass_tips, unsigned int nsubsteps, const char* format,
                                        unsigned int n_subsamples, unsigned int subsample_depth, bool subsample_with_replacement,
                                        unsigned int pcoa_dims,
                                        unsigned int permanova_perms, const char *grouping_filename, const char *grouping_columns,
                                        const char *mmap_dir)
{
    compute_status rc = okay;
    SETUP_TDBG("unifrac_multi_to_file")

    if (subsample_depth<1) {
      fprintf(stderr, "ERROR: subsampling depth cannot be 0.\n");
      return table_empty;
    }

    CHECK_FILE(biom_filename, table_missing)
    CHECK_FILE(tree_filename, tree_missing)
    PARSE_TREE_TABLE(tree_filename, biom_filename)
    TDBG_STEP("load_files")

    typedef const char* Tcstring;
    std::vector<std::string> columns;
    Tcstring *columns_c = NULL;

    // permanova values
    TReal *pm_fstats = NULL;
    TReal *pm_pvalues = NULL;
    uint32_t *pm_n_groups = NULL;
    std::vector<std::string> pm_columns;
    Tcstring *pm_columns_c = NULL;

    if (permanova_perms>0) {
         columns = stringlist_to_vector(grouping_columns);
         const unsigned int n_columns = columns.size();
         const unsigned int n_els = n_columns*n_subsamples;

         columns_c = new Tcstring[n_columns];
         for (unsigned int i=0; i<n_columns; i++)  columns_c[i] = columns[i].c_str();

         pm_fstats = new TReal[n_els];
         pm_pvalues = new TReal[n_els];
         pm_n_groups = new uint32_t[n_els];
         pm_columns.resize(n_els);
         pm_columns_c = new Tcstring[n_els];
    }

    try {
      su::WriteHDF5Multi<TReal,TMat> h5obj(real_id, out_filename,pcoa_dims);

      for (unsigned int i=0; i<n_subsamples; i++) {
        su::skbio_biom_subsampled table_subsampled(table, subsample_with_replacement, subsample_depth);
        if ((table_subsampled.n_samples==0) || (table_subsampled.n_obs==0)) {
           rc = table_empty;
           break;
        }
        TDBG_STEP("subsampled")

        TMat* result = NULL;
        rc = one_off_matrix_T<TReal,TMat>(table_subsampled,tree,unifrac_method,variance_adjust,alpha,bypass_tips,nsubsteps,mmap_dir,&result);
        if (rc!=okay) break;
        TDBG_STEP("matrix computed")

        if (permanova_perms>0) {
            const unsigned int n_columns = columns.size();
            TReal *fstats = pm_fstats+(i*n_columns);
            TReal *pvalues = pm_pvalues+(i*n_columns);
            uint32_t *n_groups = pm_n_groups+(i*n_columns);

            rc = compute_permanova_T<TReal,TMat>(grouping_filename,n_columns,columns_c,result,permanova_perms,fstats,pvalues,n_groups);
            if (rc!=okay) break;
            TDBG_STEP("permanova computed")

            char fmtstr[32];
            sprintf(fmtstr,":%d",i);
            for (unsigned int j=0; j<n_columns;j ++) {
              const unsigned int idx = i*n_columns + j;

              pm_columns[idx] = columns[j] + fmtstr;
              pm_columns_c[idx] = pm_columns[idx].c_str();
            }
        }
        h5obj.add_result(result, save_dist);
        destroy_mat_full_T<TMat,TReal>(&result);
      } // for i
      if ((rc==okay)&&(permanova_perms>0)) {
              const unsigned int n_columns = columns.size();
              const unsigned int n_els = n_columns*n_subsamples;
              std::string stat_method("PERMANOVA");
              Tcstring *stat_methods = new Tcstring[n_els];
              for (unsigned int i=0; i<n_els; i++)  stat_methods[i] = stat_method.c_str();
              std::string stat_name("pseudo-F");
              Tcstring *stat_names = new Tcstring[n_els];
              for (unsigned int i=0; i<n_els; i++)  stat_names[i] = stat_name.c_str();
              uint32_t *nperm_arr = new uint32_t[n_els];
              for (unsigned int i=0; i<n_els; i++)  nperm_arr[i] = permanova_perms;

              h5obj.write_stats(n_els, stat_methods, stat_names,
                                pm_fstats, pm_pvalues, nperm_arr,
                                pm_columns_c, pm_n_groups);

              delete[] nperm_arr;
              delete[] stat_names;
              delete[] stat_methods;
      }
    } catch (...) {
       // the only one throwing should be h5obj
       rc = output_error;
    }

    TDBG_STEP("finished")

    if (pm_columns_c!=NULL) delete[] pm_columns_c;
    if (pm_n_groups!=NULL)  delete[] pm_n_groups;
    if (pm_pvalues!=NULL)   delete[] pm_pvalues;
    if (pm_fstats!=NULL)    delete[] pm_fstats;
    if (columns_c!=NULL)    delete[] columns_c;
    return rc;
}

compute_status unifrac_multi_to_file_v2(const char* biom_filename, const char* tree_filename, const char* out_filename,
                                        const char* unifrac_method, bool variance_adjust, double alpha,
                                        bool bypass_tips, unsigned int nsubsteps, const char* format,
                                        unsigned int n_subsamples, unsigned int subsample_depth, bool subsample_with_replacement,
                                        unsigned int pcoa_dims,
                                        unsigned int permanova_perms, const char *grouping_filename, const char *grouping_columns,
                                        const char *mmap_dir)
{
    bool fp64;
    bool save_dist;
    compute_status rc = is_fp64(unifrac_method, format, fp64, save_dist);

    if (rc!=okay) {
      return rc;
    }

    if (fp64) {
      rc = unifrac_multi_to_file_T<double,mat_full_fp64_t>(H5T_IEEE_F64LE, save_dist,
                                     biom_filename, tree_filename, out_filename,
                                     unifrac_method, variance_adjust, alpha,
                                     bypass_tips, nsubsteps, format,
                                     n_subsamples, subsample_depth, subsample_with_replacement,
                                     pcoa_dims, permanova_perms, grouping_filename, grouping_columns,
                                     mmap_dir);
   } else {
      rc = unifrac_multi_to_file_T<float,mat_full_fp32_t>(H5T_IEEE_F32LE, save_dist,
                                     biom_filename, tree_filename, out_filename,
                                     unifrac_method, variance_adjust, alpha,
                                     bypass_tips, nsubsteps, format,
                                     n_subsamples, subsample_depth, subsample_with_replacement,
                                     pcoa_dims, permanova_perms, grouping_filename, grouping_columns,
                                     mmap_dir);
   }

   return rc;
}


IOStatus write_mat(const char* output_filename, mat_t* result) {
    std::ofstream output;
    output.open(output_filename);

    uint64_t comb_N = su::comb_2(result->n_samples);
    uint64_t comb_N_minus = 0;
    double v;

    for(unsigned int i = 0; i < result->n_samples; i++)
        output << "\t" << result->sample_ids[i];
    output << std::endl;

    for(unsigned int i = 0; i < result->n_samples; i++) {
        output << result->sample_ids[i];
        for(unsigned int j = 0; j < result->n_samples; j++) {
            if(i < j) { // upper triangle
                comb_N_minus = su::comb_2(result->n_samples - i);
                v = result->condensed_form[comb_N - comb_N_minus + (j - i - 1)];
            } else if (i > j) { // lower triangle
                comb_N_minus = su::comb_2(result->n_samples - j);
                v = result->condensed_form[comb_N - comb_N_minus + (i - j - 1)];
            } else {
                v = 0.0;
            }
            output << std::setprecision(16) << "\t" << v;
        }
        output << std::endl;
    }
    output.close();

    return write_okay;
}

IOStatus write_mat_from_matrix(const char* output_filename, mat_full_fp64_t* result) {
    const double *buf2d  = result->matrix;

    std::ofstream output;
    output.open(output_filename);

    double v;
    const uint64_t n_samples_64 = result->n_samples; // 64-bit to avoid overflow

    for(unsigned int i = 0; i < result->n_samples; i++)
        output << "\t" << result->sample_ids[i];
    output << std::endl;

    for(unsigned int i = 0; i < result->n_samples; i++) {
        output << result->sample_ids[i];
        for(unsigned int j = 0; j < result->n_samples; j++) {
            v = buf2d[i*n_samples_64+j];
            output << std::setprecision(16) << "\t" << v;
        }
        output << std::endl;
    }
    output.close();

    return write_okay;
}

// Internal: Make sure TReal and real_id match
template<class TReal, class TMat>
inline IOStatus write_mat_from_matrix_hdf5_T(const char* output_filename, TMat * result, hid_t real_id,
                                             unsigned int pcoa_dims, bool save_dist,
                                             unsigned int           stat_n_vals,
                                             const char* const    * stat_method_arr, const char* const  * stat_name_arr,
                                             const TReal          * stat_val_arr,    const TReal        * stat_pval_arr, const uint32_t  * stat_perm_count_arr,
                                             const char* const    * stat_group_name_arr, const uint32_t * stat_group_count_arr) {
   SETUP_TDBG("write_mat_from_matrix")
   /* Create a new file using default properties. */
   hid_t output_file_id = H5Fcreate(output_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   if (output_file_id<0) return write_error;

   const auto n_samples = result->n_samples;

   // simple header
   if (write_hdf5_string(output_file_id,"format","BDSM")<0) {
       H5Fclose (output_file_id);
       return write_error;
   }
   if (write_hdf5_string(output_file_id,"version","2020.12")<0) {
       H5Fclose (output_file_id);
       return write_error;
   }

   // save the ids
   {
     herr_t status = write_hdf5_stringarray(output_file_id,
                         "order", n_samples, result->sample_ids);
     if (status<0) {
       H5Fclose (output_file_id);
       return write_error;
     }
   }
   TDBG_STEP("header saved")

   // save the matrix
   if (save_dist) {
     herr_t status = write_hdf5_array2D<TReal>(output_file_id,real_id,
                         "matrix", n_samples, n_samples, result->matrix);
     if (status<0) {
       H5Fclose (output_file_id);
       return write_error;
     }
     TDBG_STEP("matrix saved")
   }

   if (pcoa_dims>0) {
     // compute pcoa and save it in the file
     // use inplace variant to keep memory use in check; we don't need matrix anymore
     TReal * eigenvalues;
     TReal * samples;
     TReal * proportion_explained;

     su::pcoa_inplace(result->matrix, n_samples, pcoa_dims, eigenvalues, samples, proportion_explained);
     TDBG_STEP("pcoa computed")


     IOStatus rc = write_okay;
     if (write_hdf5_string(output_file_id,"pcoa_method","FSVD")<0) {
       rc = write_error;
     } else {
       rc = append_hdf5_pcoa(output_file_id, real_id, pcoa_dims, n_samples,
                             "pcoa_eigvals", "pcoa_samples", "pcoa_proportion_explained",
                             eigenvalues, samples,  proportion_explained);
     }
     free(eigenvalues);
     free(proportion_explained);
     free(samples);
     if (rc!=write_okay) {
       H5Fclose (output_file_id);
       return rc;
     }
     TDBG_STEP("pcoa saved")
   } // if pcoa

   if (stat_n_vals>0) {
     herr_t status = write_hdf5_stringarray(output_file_id,
                         "stat_methods", stat_n_vals, stat_method_arr);
     if (status>=0) {
       status = write_hdf5_stringarray(output_file_id,
                         "stat_test_names", stat_n_vals, stat_name_arr);
     }
     if (status>=0) {
       status = write_hdf5_stringarray(output_file_id,
                         "stat_grouping_names", stat_n_vals, stat_group_name_arr);
     }
     if (status>=0) {
       status = write_hdf5_array<uint32_t>(output_file_id, H5T_STD_U32LE,
                         "stat_n_groups", stat_n_vals, stat_group_count_arr);
     }
     if (status>=0) {
       status = write_hdf5_array<TReal>(output_file_id,real_id,
                         "stat_values", stat_n_vals, stat_val_arr);
     }
     if (status>=0) {
       status = write_hdf5_array<TReal>(output_file_id,real_id,
                         "stat_pvalues", stat_n_vals, stat_pval_arr);
     }
     if (status>=0) {
       status = write_hdf5_array<uint32_t>(output_file_id, H5T_STD_U32LE,
                         "stat_n_permutations", stat_n_vals, stat_perm_count_arr);
     }
     
     // check status after cleanup, for simplicity
     if (status<0) {
       H5Fclose (output_file_id);
       return write_error;
     }
     TDBG_STEP("stats saved")
   } // if stat_n_vals

   H5Fclose (output_file_id);
   return write_okay;
}

// Internal: Make sure TReal and real_id match
// Note: Deprecated, for backwards compatibility only
template<class TReal, class TMat>
inline IOStatus write_mat_hdf5_T(const char* output_filename, mat_t* result,hid_t real_id, unsigned int pcoa_dims, bool save_dist) {
     // compute the matrix
     TMat mat_full;
     mat_full.n_samples = result->n_samples;

     const uint64_t n_samples = result->n_samples;
     mat_full.flags = 0;
     mat_full.matrix = (TReal*) malloc(n_samples*n_samples*sizeof(TReal));
     if (mat_full.matrix==NULL) {
       return open_error; // we don't have a better error code
     }

     mat_full.sample_ids = result->sample_ids; // just link

     condensed_form_to_matrix_T(result->condensed_form, n_samples, mat_full.matrix);
     IOStatus err =  write_mat_from_matrix_hdf5_T<TReal,TMat>(output_filename, &mat_full, real_id, pcoa_dims, save_dist,
                             0,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

     free(mat_full.matrix);
     return err;
}

// Note: Deprecated, for backwards compatibility only
IOStatus write_mat_hdf5_fp64(const char* output_filename, mat_t* result, unsigned int pcoa_dims, int save_dist) {
  return write_mat_hdf5_T<double,mat_full_fp64_t>(output_filename,result,H5T_IEEE_F64LE,pcoa_dims,save_dist);
}

// Note: Deprecated, for backwards compatibility only
IOStatus write_mat_hdf5_fp32(const char* output_filename, mat_t* result, unsigned int pcoa_dims, int save_dist) {
  return write_mat_hdf5_T<float,mat_full_fp32_t>(output_filename,result,H5T_IEEE_F32LE,pcoa_dims,save_dist);
}

IOStatus write_mat_from_matrix_hdf5_fp64_v2(const char* output_filename, mat_full_fp64_t* result,
                                            unsigned int pcoa_dims, int save_dist,
                                            unsigned int stat_n_vals,
                                            const char*  *stat_method_arr,     const char*        *stat_name_arr,
                                            const double *stat_val_arr,        const double       *stat_pval_arr, const unsigned int *stat_perm_count_arr,
                                            const char*  *stat_group_name_arr, const unsigned int *stat_group_count_arr) {
  return write_mat_from_matrix_hdf5_T<double,mat_full_fp64_t>(output_filename,result,H5T_IEEE_F64LE,pcoa_dims,save_dist,
                        stat_n_vals,stat_method_arr,stat_name_arr,stat_val_arr,stat_pval_arr,stat_perm_count_arr,stat_group_name_arr,stat_group_count_arr);
}

IOStatus write_mat_from_matrix_hdf5_fp32_v2(const char* output_filename, mat_full_fp32_t* result,
                                            unsigned int pcoa_dims, int save_dist,
                                            unsigned int stat_n_vals,
                                            const char*  *stat_method_arr,     const char*        *stat_name_arr,
                                            const float  *stat_val_arr,        const float        *stat_pval_arr, const unsigned int *stat_perm_count_arr,
                                            const char*  *stat_group_name_arr, const unsigned int *stat_group_count_arr) {
  return write_mat_from_matrix_hdf5_T<float,mat_full_fp32_t>(output_filename,result,H5T_IEEE_F32LE,pcoa_dims,save_dist,
                        stat_n_vals,stat_method_arr,stat_name_arr,stat_val_arr,stat_pval_arr,stat_perm_count_arr,stat_group_name_arr,stat_group_count_arr);
}

// Backwards compatibility
IOStatus write_mat_from_matrix_hdf5_fp64(const char* output_filename, mat_full_fp64_t* result, unsigned int pcoa_dims, int save_dist) {
  return write_mat_from_matrix_hdf5_fp64_v2(output_filename,result,pcoa_dims,save_dist,
                             0,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
}

// Backwards compatibility
IOStatus write_mat_from_matrix_hdf5_fp32(const char* output_filename, mat_full_fp32_t* result, unsigned int pcoa_dims, int save_dist) {
  return write_mat_from_matrix_hdf5_fp32_v2(output_filename,result,pcoa_dims,save_dist,
                             0,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
}

IOStatus write_vec(const char* output_filename, r_vec* result) {
    std::ofstream output;
    output.open(output_filename);

    // write sample ids in first column of file and faith's pd in second column
    output << "#SampleID\tfaith_pd" << std::endl;
    for(unsigned int i = 0; i < result->n_samples; i++) {
        output << result->sample_ids[i];
        output << std::setprecision(16) << "\t" << result->values[i];
        output << std::endl;
    }
    output.close();

    return write_okay;
}

IOStatus write_partial(const char* output_filename, const partial_mat_t* result) {
    int fd = open(output_filename, O_WRONLY | O_CREAT | O_TRUNC,  S_IRUSR |  S_IWUSR );
    if (fd==-1) return write_error;

    int cnt = -1;

    uint32_t n_stripes = result->stripe_stop - result->stripe_start;

    uint32_t sample_id_length = 0;
    for(unsigned int i = 0; i < result->n_samples; i++) {
        sample_id_length += strlen(result->sample_ids[i])+1;
    }

    {
      char * const samples_buf = (char *)malloc(sample_id_length);
 
      char *samples_ptr = samples_buf;

      /* sample IDs */
      for(unsigned int i = 0; i < result->n_samples; i++) {
          uint32_t length = strlen(result->sample_ids[i])+1;
          memcpy(samples_ptr,result->sample_ids[i],length);
          samples_ptr+= length;
      }

      int max_compressed = LZ4_compressBound(sample_id_length);
      char * const cmp_buf = (char *)malloc(max_compressed);

      int sample_id_length_compressed = LZ4_compress_default(samples_buf,cmp_buf,sample_id_length,max_compressed);
      if (sample_id_length_compressed<1)  {close(fd); return open_error;}

      uint32_t header[8];
      header[0] = PARTIAL_MAGIC_V2;
      header[1] = result->n_samples;
      header[2] = n_stripes;
      header[3] = result->stripe_start;
      header[4] = result->stripe_total;
      header[5] = result->is_upper_triangle;
      header[6] = sample_id_length;
      header[7] = sample_id_length_compressed;

      cnt=write(fd,header, 8 * sizeof(uint32_t));
      if (cnt<1)  {close(fd); return write_error;}

      cnt=write(fd,cmp_buf, sample_id_length_compressed);
      if (cnt<1)  {close(fd); return write_error;}

      free(cmp_buf);
      free(samples_buf);
    }

    {
      int max_compressed = LZ4_compressBound(sizeof(double) * result->n_samples);
      char * const cmp_buf_raw = (char *)malloc(max_compressed+sizeof(uint32_t));
      char * const cmp_buf = cmp_buf_raw + sizeof(uint32_t);

      /* stripe information */
      for(unsigned int i = 0; i < n_stripes; i++) {
        int cmp_size = LZ4_compress_default((const char *) result->stripes[i],cmp_buf,sizeof(double) * result->n_samples,max_compressed);
        if (cmp_size<1)  {close(fd); return open_error;}

        uint32_t *cmp_buf_size_p = (uint32_t *)cmp_buf_raw;
        *cmp_buf_size_p = cmp_size;

        cnt=write(fd, cmp_buf_raw, cmp_size+sizeof(uint32_t));
        if (cnt<1) {return write_error;}
      }

      free(cmp_buf_raw);
    }

    /* footer */
    {
      uint32_t header[1];
      header[0] = PARTIAL_MAGIC_V2;

      cnt=write(fd,header, 1 * sizeof(uint32_t));
      if (cnt<1)  {close(fd); return open_error;}
    }

    close(fd);

    return write_okay;
}

IOStatus _is_partial_file(const char* input_filename) {
    int fd = open(input_filename, O_RDONLY );
    if (fd==-1) return open_error;

    uint32_t header[1];
    int cnt = read(fd,header,sizeof(uint32_t));
    close(fd);

    if (cnt!=sizeof(uint32_t)) return magic_incompatible;
    if ( header[0] != PARTIAL_MAGIC_V2) return magic_incompatible;

    return read_okay;
}

template<class TPMat>
inline IOStatus read_partial_header_fd(int fd, TPMat &result) {
    int cnt=-1;

    uint32_t header[8];
    cnt = read(fd,header,8*sizeof(uint32_t));
    if (cnt != (8*sizeof(uint32_t))) {return magic_incompatible;}

    if ( header[0] != PARTIAL_MAGIC_V2) {return magic_incompatible;}

    const uint32_t n_samples = header[1];
    const uint32_t n_stripes = header[2];
    const uint32_t stripe_start = header[3];
    const uint32_t stripe_total = header[4];
    const bool is_upper_triangle = header[5];

    /* sanity check header */
    if(n_samples <= 0 || n_stripes <= 0 || stripe_total <= 0 || is_upper_triangle < 0)
         {return bad_header;}
    if(stripe_total >= n_samples || n_stripes > stripe_total || stripe_start >= stripe_total || stripe_start + n_stripes > stripe_total)
         {return bad_header;}

    /* initialize the partial result structure */
    result.n_samples = n_samples;
    result.sample_ids = (char**)malloc(sizeof(char*) * n_samples);
    result.stripes = (double**)malloc(sizeof(double*) * (n_stripes));
    result.stripe_start = stripe_start;
    result.stripe_stop = stripe_start + n_stripes;
    result.is_upper_triangle = is_upper_triangle;
    result.stripe_total = stripe_total;

    /* load samples */
    {
      const uint32_t sample_id_length = header[6];
      const uint32_t sample_id_length_compressed = header[7];

      /* sanity check header */
      if (sample_id_length<=0 || sample_id_length_compressed <=0)
         { return bad_header;}

      char * const cmp_buf = (char *)malloc(sample_id_length_compressed);
      if (cmp_buf==NULL) { return bad_header;} // no better error code
      cnt = read(fd,cmp_buf,sample_id_length_compressed);
      if (cnt != sample_id_length_compressed) {free(cmp_buf); return magic_incompatible;}

      char *samples_buf = (char *)malloc(sample_id_length);
      if (samples_buf==NULL) { free(cmp_buf); return bad_header;} // no better error code

      cnt = LZ4_decompress_safe(cmp_buf,samples_buf,sample_id_length_compressed,sample_id_length);
      if (cnt!=sample_id_length) {free(samples_buf); free(cmp_buf); return magic_incompatible;}

      const char *samples_ptr = samples_buf;

      for(int i = 0; i < n_samples; i++) {
        uint32_t sample_length = strlen(samples_ptr);
        if ((samples_ptr+sample_length+1)>(samples_buf+sample_id_length)) {free(samples_buf); free(cmp_buf); return magic_incompatible;}

        result.sample_ids[i] = (char*)malloc(sample_length + 1);
        memcpy(result.sample_ids[i],samples_ptr,sample_length + 1);
        samples_ptr += sample_length + 1;
      }
      free(samples_buf);
      free(cmp_buf);
    }

    return read_okay;
}

template<class TPMat>
inline IOStatus read_partial_data_fd(int fd, TPMat &result) {
    int cnt=-1;

    const uint32_t n_samples = result.n_samples;
    const uint32_t n_stripes = result.stripe_stop-result.stripe_start;

    /* load stripes */
    {
      int max_compressed = LZ4_compressBound(sizeof(double) * n_samples);
      char * const cmp_buf = (char *)malloc(max_compressed+sizeof(uint32_t));
      if (cmp_buf==NULL) { return bad_header;} // no better error code

      uint32_t *cmp_buf_size_p = (uint32_t *)cmp_buf;

      cnt = read(fd,cmp_buf_size_p , sizeof(uint32_t) );
      if (cnt != sizeof(uint32_t) ) {free(cmp_buf); return magic_incompatible;}

      for(int i = 0; i < n_stripes; i++) {
        uint32_t cmp_size = *cmp_buf_size_p;

        uint32_t read_size = cmp_size;
        if ( (i+1)<n_stripes ) read_size += sizeof(uint32_t); // last one does not have the cmp_size

        cnt = read(fd,cmp_buf , read_size );
        if (cnt != read_size) {free(cmp_buf); return magic_incompatible;}

        result.stripes[i] = (double *) malloc(sizeof(double) * n_samples);
        if(result.stripes[i] == NULL) {
            fprintf(stderr, "failed\n");
            exit(1);
        }
        cnt = LZ4_decompress_safe(cmp_buf, (char *) result.stripes[i],cmp_size,sizeof(double) * n_samples);
        if (cnt != ( sizeof(double) * n_samples ) ) {free(cmp_buf); return magic_incompatible;}

        cmp_buf_size_p = (uint32_t *)(cmp_buf+cmp_size);
      }

      free(cmp_buf);
    }

    return read_okay;
}

template<class TPMat>
inline IOStatus read_partial_one_stripe_fd(int fd, TPMat &result, uint32_t stripe_idx) {
    int cnt=-1;

    const uint32_t n_samples = result.n_samples;

    /* load stripes */
    {
      int max_compressed = LZ4_compressBound(sizeof(double) * n_samples);
      char * const cmp_buf = (char *)malloc(max_compressed+sizeof(uint32_t));
      if (cmp_buf==NULL) { return bad_header;} // no better error code

      uint32_t *cmp_buf_size_p = (uint32_t *)cmp_buf;

      uint32_t curr_idx = stripe_idx;
      while (result.offsets[curr_idx]==0) --curr_idx; // must start reading from the first known offset

      for (;curr_idx<stripe_idx; curr_idx++) { // now get all the intermediate indexes
        if (lseek(fd, result.offsets[curr_idx], SEEK_SET)!=result.offsets[curr_idx]) {
           free(cmp_buf); return bad_header;
        }

        cnt = read(fd,cmp_buf_size_p , sizeof(uint32_t) );
        if (cnt != sizeof(uint32_t) ) {free(cmp_buf); return magic_incompatible;}

        uint32_t cmp_size = *cmp_buf_size_p;
        uint32_t read_size = cmp_size;

        result.offsets[curr_idx+1] = result.offsets[curr_idx] + sizeof(uint32_t) + read_size;
      }

      // =======================
      // ready to read my stripe

      if (lseek(fd, result.offsets[stripe_idx], SEEK_SET)!=result.offsets[stripe_idx]) {
         free(cmp_buf); return bad_header;
      }

      cnt = read(fd,cmp_buf_size_p , sizeof(uint32_t) );
      if (cnt != sizeof(uint32_t) ) {free(cmp_buf); return magic_incompatible;}

      {
        uint32_t cmp_size = *cmp_buf_size_p;

        uint32_t read_size = cmp_size;

        cnt = read(fd,cmp_buf , read_size );
        if (cnt != read_size) {free(cmp_buf); return magic_incompatible;}

        result.stripes[stripe_idx] = (double *) malloc(sizeof(double) * n_samples);
        if(result.stripes[stripe_idx] == NULL) {
            fprintf(stderr, "failed\n");
            exit(1);
        }
        cnt = LZ4_decompress_safe(cmp_buf, (char *) result.stripes[stripe_idx],cmp_size,sizeof(double) * n_samples);
        if (cnt != ( sizeof(double) * n_samples ) ) {free(cmp_buf); return magic_incompatible;}
      }

      free(cmp_buf);
    }

    return read_okay;
}

IOStatus read_partial(const char* input_filename, partial_mat_t** result_out) {
    int fd = open(input_filename, O_RDONLY );
    if (fd==-1) return open_error;

    /* initialize the partial result structure */
    partial_mat_t* result = (partial_mat_t*)malloc(sizeof(partial_mat));

    IOStatus sts = magic_incompatible;

    sts = read_partial_header_fd<partial_mat_t>(fd, *result);
    if (sts==read_okay)
       sts = read_partial_data_fd<partial_mat_t>(fd, *result);

    if (sts==read_okay) {
      IOStatus sts = read_okay;
      /* sanity check the footer */
      uint32_t header[1];
      header[0] = 0;
      int cnt = read(fd,header,sizeof(uint32_t));
      if (cnt != (sizeof(uint32_t))) {sts= magic_incompatible;}
    
      if (sts==read_okay) {
        if ( header[0] != PARTIAL_MAGIC_V2) {sts= magic_incompatible;}
      }
    }

    close(fd);

    if (sts==read_okay) {
      (*result_out) = result;
    } else {
      free(result);
      (*result_out) = NULL;
    }
    return sts;
}

IOStatus read_partial_header(const char* input_filename, partial_dyn_mat_t** result_out) {
    int fd = open(input_filename, O_RDONLY );
    if (fd==-1) return open_error;

    /* initialize the partial result structure */
    partial_dyn_mat_t* result = (partial_dyn_mat_t*)malloc(sizeof(partial_dyn_mat));
    {
      IOStatus sts = read_partial_header_fd<partial_dyn_mat_t>(fd, *result);
      if (sts!=read_okay) {free(result); close(fd); return sts;}
    }

    // save the offset of the first stripe
    const uint32_t n_stripes = result->stripe_stop-result->stripe_start;
    result->stripes = (double**) calloc(n_stripes,sizeof(double*));
    result->offsets = (uint64_t*) calloc(n_stripes,sizeof(uint64_t));
    result->offsets[0] = lseek(fd,0,SEEK_CUR);
    
    close(fd);

    result->filename= strdup(input_filename);

    (*result_out) = result;
    return read_okay;
}

IOStatus read_partial_one_stripe(partial_dyn_mat_t* result, uint32_t stripe_idx) {
    if (result->stripes[stripe_idx]!=0) return read_okay; // will not re-read

    int fd = open(result->filename, O_RDONLY );
    if (fd==-1) return open_error;

    IOStatus sts = read_partial_one_stripe_fd<partial_dyn_mat_t>(fd, *result, stripe_idx);

    close(fd);
    return sts;
}


template<class TPMat>
MergeStatus check_partial(const TPMat* const * partial_mats, int n_partials, bool verbose) {
    if(n_partials <= 0) {
        fprintf(stderr, "Zero or less partials.\n");
        exit(EXIT_FAILURE);
    }

    // sanity check
    int n_samples = partial_mats[0]->n_samples;
    bool *stripe_map = (bool*)calloc(sizeof(bool), partial_mats[0]->stripe_total);
    int stripe_count = 0;

    for(int i = 0; i < n_partials; i++) {
        if(partial_mats[i]->n_samples != n_samples) {
            free(stripe_map);
            if (verbose) {
                fprintf(stderr, "Wrong number of samples in %i, %i!=%i\n",
                        i,int(partial_mats[i]->n_samples),int(n_samples));
            }
            return partials_mismatch;
        }

        if(partial_mats[0]->stripe_total != partial_mats[i]->stripe_total) {
            free(stripe_map);
            if (verbose) {
                fprintf(stderr, "Wrong number of stripes in %i, %i!=%i\n",
                        i,int(partial_mats[0]->stripe_total), int(partial_mats[i]->stripe_total));
            }
            return partials_mismatch;
        }
        if(partial_mats[0]->is_upper_triangle != partial_mats[i]->is_upper_triangle) {
            free(stripe_map);
            if (verbose) {
                fprintf(stderr, "Wrong number of is_upper_triangle in %i, %i!=%i\n",
                        i,int(partial_mats[0]->is_upper_triangle),int(partial_mats[i]->is_upper_triangle));
            }
            return square_mismatch;
        }
        for(int j = 0; j < n_samples; j++) {
            if(strcmp(partial_mats[0]->sample_ids[j], partial_mats[i]->sample_ids[j]) != 0) {
                free(stripe_map);
                if (verbose) {
                    fprintf(stderr, "Wrong number of sample id %i in %i, %s!=%s\n",
                            j,i,partial_mats[0]->sample_ids[j], partial_mats[i]->sample_ids[j]);
                }
                return sample_id_consistency;
            }
        }
        for(int j = partial_mats[i]->stripe_start; j < partial_mats[i]->stripe_stop; j++) {
            if(stripe_map[j]) {
                if (verbose) {
                    fprintf(stderr, "Overlap in %i vs %i\n",
                            i,j);
                }
                free(stripe_map);
                return stripes_overlap;
            }
            stripe_map[j] = true;
            stripe_count += 1;
        }
    }
    free(stripe_map);

    if(stripe_count != partial_mats[0]->stripe_total) {
        if (verbose) {
            fprintf(stderr, "Insufficient number of stripes found, %i!=%i\n",
                    int(stripe_count), int(partial_mats[0]->stripe_total));
        }
        return incomplete_stripe_set;
    }

    return merge_okay;
}

MergeStatus validate_partial(const partial_dyn_mat_t* const * partial_mats, int n_partials) {
    return check_partial(partial_mats, n_partials, true);
}


MergeStatus merge_partial(partial_mat_t** partial_mats, int n_partials, unsigned int dummy, mat_t** result) {
    MergeStatus err = check_partial(partial_mats, n_partials, false);
    if (err!=merge_okay) return err;

    int n_samples = partial_mats[0]->n_samples;
    std::vector<double*> stripes(partial_mats[0]->stripe_total);
    std::vector<double*> stripes_totals(partial_mats[0]->stripe_total);  // not actually used but destroy_stripes needs this to "exist"
    for(int i = 0; i < n_partials; i++) {
        int n_stripes = partial_mats[i]->stripe_stop - partial_mats[i]->stripe_start;
        for(int j = 0; j < n_stripes; j++) {
            // as this is potentially a large amount of memory, don't copy, just adopt
            *&(stripes[j + partial_mats[i]->stripe_start]) = partial_mats[i]->stripes[j];
        }
    }

    initialize_mat_no_biom(*result, partial_mats[0]->sample_ids, n_samples, partial_mats[0]->is_upper_triangle);
    if ((*result)==NULL) return incomplete_stripe_set;
    if ((*result)->condensed_form==NULL) return incomplete_stripe_set;
    if ((*result)->sample_ids==NULL) return incomplete_stripe_set;

    su::stripes_to_condensed_form(stripes, n_samples, (*result)->condensed_form, 0, partial_mats[0]->stripe_total);

    destroy_stripes(stripes, stripes_totals, n_samples, 0, n_partials);

    return merge_okay;
}

// Will keep only the strictly necessary stripes in memory... reading just in time
class PartialStripes : public su::ManagedStripes {
        private:
           const uint32_t n_partials;
           mutable partial_dyn_mat_t* * partial_mats; // link only, not owned

           static bool in_range(const partial_dyn_mat_t &partial_mat, uint32_t stripe) {
             return (stripe>=partial_mat.stripe_start) && (stripe<partial_mat.stripe_stop);
           }

           uint32_t find_partial_idx(uint32_t stripe) const {
              for (uint32_t i=0; i<n_partials; i++) {
                if (in_range(*(partial_mats[i]),stripe)) return i;
              }
              return 0; // should never get here
           }
        public:
           PartialStripes(uint32_t _n_partials, partial_dyn_mat_t* * _partial_mats)
           : n_partials(_n_partials)
           , partial_mats(_partial_mats)
           {}

           virtual const double *get_stripe(uint32_t stripe) const {
              uint32_t pidx = find_partial_idx(stripe);
              partial_dyn_mat_t * const partial_mat = partial_mats[pidx];
              uint32_t sidx = stripe-partial_mat->stripe_start;

              if (partial_mat->stripes[sidx]==NULL) {
                  read_partial_one_stripe(partial_mat,sidx);
                  // ignore any errors, not clear what to do
                  // will just return NULL
              }

              return partial_mat->stripes[sidx];
           }
           virtual void release_stripe(uint32_t stripe) const {
              uint32_t pidx = find_partial_idx(stripe);
              partial_dyn_mat_t * const partial_mat = partial_mats[pidx];
              uint32_t sidx = stripe-partial_mat->stripe_start;

              if (partial_mat->stripes[sidx]!=NULL) {
                 free(partial_mat->stripes[sidx]);
                 partial_mat->stripes[sidx]=NULL;
              }
           }
};

template<class TReal, class TMat>
MergeStatus merge_partial_to_matrix_T(partial_dyn_mat_t* * partial_mats, int n_partials, 
                                      const char *mmap_dir, /* if NULL or "", use malloc */
                                      TMat** result /* out */ ) {
    if (mmap_dir!=NULL) {
     if (mmap_dir[0]==0) mmap_dir = NULL; // easier to have a simple test going on
    }

    MergeStatus err = check_partial(partial_mats, n_partials, false);
    if (err!=merge_okay) return err;

    initialize_mat_full_no_biom_T<TReal,TMat>(*result, partial_mats[0]->sample_ids, partial_mats[0]->n_samples,mmap_dir);

    if ((*result)==NULL) return incomplete_stripe_set;
    if ((*result)->matrix==NULL) return incomplete_stripe_set;
    if ((*result)->sample_ids==NULL) return incomplete_stripe_set;

    PartialStripes ps(n_partials,partial_mats);
    const uint32_t tile_size = (mmap_dir==NULL) ? \
                                  (128/sizeof(TReal)) : /* keep it small for memory access, to fit in chip cache */ \
                                  (4096/sizeof(TReal)); /* make it larger for mmap, as the limiting factor is swapping */
    su::stripes_to_matrix_T<TReal>(ps, partial_mats[0]->n_samples, partial_mats[0]->stripe_total, (*result)->matrix, tile_size);

    return merge_okay;
}

MergeStatus merge_partial_to_matrix(partial_dyn_mat_t* * partial_mats, int n_partials, mat_full_fp64_t** result) {
  return merge_partial_to_matrix_T<double,mat_full_fp64_t>(partial_mats, n_partials, NULL, result);
}

MergeStatus merge_partial_to_matrix_fp32(partial_dyn_mat_t* * partial_mats, int n_partials, mat_full_fp32_t** result) {
  return merge_partial_to_matrix_T<float,mat_full_fp32_t>(partial_mats, n_partials, NULL, result);
}

MergeStatus merge_partial_to_mmap_matrix(partial_dyn_mat_t* * partial_mats, int n_partials, const char *mmap_dir, mat_full_fp64_t** result) {
  return merge_partial_to_matrix_T<double,mat_full_fp64_t>(partial_mats, n_partials, mmap_dir, result);
}

MergeStatus merge_partial_to_mmap_matrix_fp32(partial_dyn_mat_t* * partial_mats, int n_partials, const char *mmap_dir, mat_full_fp32_t** result) {
  return merge_partial_to_matrix_T<float,mat_full_fp32_t>(partial_mats, n_partials, mmap_dir, result);
}


// skbio_alt pass-thoughs


// Find eigen values and vectors
// Based on N. Halko, P.G. Martinsson, Y. Shkolnisky, and M. Tygert.
//     Original Paper: https://arxiv.org/abs/1007.5510
// centered == n x n, must be symmetric, Note: will be used in-place as temp buffer

void find_eigens_fast(const uint32_t n_samples, const uint32_t n_dims, double * centered, double **eigenvalues, double **eigenvectors) {
  su::find_eigens_fast(n_samples, n_dims, centered, *eigenvalues, *eigenvectors);
}

void find_eigens_fast_fp32(const uint32_t n_samples, const uint32_t n_dims, float * centered, float **eigenvalues, float **eigenvectors) {
  su::find_eigens_fast(n_samples, n_dims, centered, *eigenvalues, *eigenvectors);
}

/*
    Perform Principal Coordinate Analysis.

    Principal Coordinate Analysis (PCoA) is a method similar
    to Principal Components Analysis (PCA) with the difference that PCoA
    operates on distance matrices, typically with non-euclidian and thus
    ecologically meaningful distances like UniFrac in microbiome research.

    In ecology, the euclidean distance preserved by Principal
    Component Analysis (PCA) is often not a good choice because it
    deals poorly with double zeros (Species have unimodal
    distributions along environmental gradients, so if a species is
    absent from two sites at the same site, it can't be known if an
    environmental variable is too high in one of them and too low in
    the other, or too low in both, etc. On the other hand, if an
    species is present in two sites, that means that the sites are
    similar.).

    Note that the returned eigenvectors are not normalized to unit length.
*/

// mat       - in, result of unifrac compute
// n_samples - in, size of the matrix (n x n)
// n_dims    - in, Dimensions to reduce the distance matrix to. This number determines how many eigenvectors and eigenvalues will be returned.
// eigenvalues - out, alocated buffer of size n_dims
// samples     - out, alocated buffer of size n_dims x n_samples
// proportion_explained - out, allocated buffer of size n_dims

void pcoa(const double * mat, const uint32_t n_samples, const uint32_t n_dims, double * *eigenvalues, double * *samples, double * *proportion_explained) {
  su::pcoa(mat, n_samples, n_dims, *eigenvalues, *samples, *proportion_explained);
}

void pcoa_fp32(const float * mat, const uint32_t n_samples, const uint32_t n_dims, float * *eigenvalues, float * *samples, float * *proportion_explained) {
  su::pcoa(mat, n_samples, n_dims, *eigenvalues, *samples, *proportion_explained);
}

void pcoa_mixed(const double * mat, const uint32_t n_samples, const uint32_t n_dims, float * *eigenvalues, float * *samples, float * *proportion_explained) {
  su::pcoa(mat, n_samples, n_dims, *eigenvalues, *samples, *proportion_explained);
}

