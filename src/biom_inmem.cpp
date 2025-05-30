/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2016-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include "biom_inmem.hpp"

using namespace su;

template<class TData>
inline static TData *malloc_wcheck(unsigned int cnt) { 
    size_t bufsize = sizeof(TData) * cnt;
    TData *my = (TData *)malloc(bufsize);
    if(my == NULL) {
        fprintf(stderr, "Failed to allocate %ld bytes; [%s]:%d\n", 
                bufsize, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
   return my;
}

template<class TData>
inline static TData *malloc_and_copy(unsigned int cnt, const TData *other) { 
    TData *my = malloc_wcheck<TData>(cnt);
    size_t bufsize = sizeof(TData) * cnt;
    std::memcpy(my, other, bufsize);
    return my;
}

static inline uint32_t count_filtered_samples(uint32_t n_samples, const double sample_counts[], const double min_sample_counts) {
   uint32_t cnt = 0;
   for (uint32_t i=0; i<n_samples; i++) {
      if (sample_counts[i]>=min_sample_counts) cnt++;
   }
   return cnt;
}

sparse_data::sparse_data(bool _clean_on_destruction) 
  : n_obs(0)
  , n_samples(0)
  , clean_on_destruction(_clean_on_destruction)
  , obs_indices_resident(NULL)
  , obs_data_resident(NULL)
  , obs_counts_resident(NULL)
{}

sparse_data::sparse_data(const sparse_data& other, bool _clean_on_destruction)
  : n_obs(other.n_obs)
  , n_samples(other.n_samples)
  , clean_on_destruction(_clean_on_destruction)
  , obs_indices_resident(_clean_on_destruction?NULL:other.obs_indices_resident)
  , obs_data_resident(_clean_on_destruction?NULL:other.obs_data_resident)
  , obs_counts_resident(_clean_on_destruction?NULL:other.obs_counts_resident)
{
    if (_clean_on_destruction && (n_obs>0)) { // must make a copy
        malloc_resident();
        for (uint32_t i = 0; i < n_obs; i++) {
            const unsigned int cnt = other.obs_counts_resident[i];
            obs_counts_resident[i]  = cnt;
            obs_data_resident[i]    = malloc_and_copy<double>(cnt, other.obs_data_resident[i]);
            obs_indices_resident[i] = malloc_and_copy<uint32_t>(cnt, other.obs_indices_resident[i]);
        }
    }
}

sparse_data::sparse_data(const sparse_data& other, const double sample_counts[], const double min_sample_counts)
  : n_obs(other.n_obs)
  , n_samples(count_filtered_samples(other.n_samples,sample_counts,min_sample_counts))
  , clean_on_destruction(true)
  , obs_indices_resident(NULL)
  , obs_data_resident(NULL)
  , obs_counts_resident(NULL)
{
    if ((n_obs>0) && (n_samples>0)) {
        // since we are not copying all element, we will need to scale the indeces
        // compute once for all of them
        uint32_t * __restrict__ count_diffs = new uint32_t[other.n_samples];
        {
           uint32_t curr_diff = 0;
           for (uint32_t i = 0; i < other.n_samples; i++) {
              if (sample_counts[i]<min_sample_counts) curr_diff++;
              count_diffs[i] = curr_diff;
           }
        }

        malloc_resident();
        for(unsigned int i = 0; i < other.n_obs; i++) {
          const unsigned int cnt = other.count_filtered_els(i, sample_counts,min_sample_counts);
          obs_counts_resident[i]  = cnt;
          if (cnt>0) {
            obs_data_resident[i]    = malloc_wcheck<double>(cnt);
            obs_indices_resident[i] = malloc_wcheck<uint32_t>(cnt);
            {
               // explicitly mark restrict to allow for compiler optimization
               double   * __restrict__ my_data    = obs_data_resident[i];
               uint32_t * __restrict__ my_indices = obs_indices_resident[i];
               const double   * __restrict__ other_data    = other.obs_data_resident[i];
               const uint32_t * __restrict__ other_indices = other.obs_indices_resident[i];
               const unsigned int other_cnt = other.obs_counts_resident[i];
               uint32_t j_cnt = 0;
               for (unsigned int j = 0; j < other_cnt; j++) {
                  const uint32_t el_idx = other_indices[j];
                  if (sample_counts[el_idx]>=min_sample_counts) {
                     my_data[j_cnt] = other_data[j];
                     // we did not copy all elements, so we need to scale the indices
                     my_indices[j_cnt] = other_indices[j]-count_diffs[el_idx];
                     j_cnt++;
                  }
               }
            }
          } else {
            obs_data_resident[i]    = NULL;
            obs_indices_resident[i] = NULL;
          }
        }
        delete[] count_diffs;
    }
}


// not using const on indices/indptr/data as the pointers are being borrowed
sparse_data::sparse_data(const uint32_t _n_obs,
                         const uint32_t _n_samples,
                         uint32_t* indices,
                         uint32_t* indptr,
                         double* data)
  : n_obs(_n_obs)
  , n_samples(_n_samples)
  , clean_on_destruction(false)
  , obs_indices_resident(NULL)
  , obs_data_resident(NULL)
  , obs_counts_resident(NULL) {

    malloc_resident();

    #pragma omp parallel for schedule(static)
    for(unsigned int i = 0; i < n_obs; i++)  {
        int32_t start = indptr[i];
        int32_t end = indptr[i + 1];
        unsigned int count = end - start;

        uint32_t* index_ptr = (indices + start);
        double* data_ptr = (data + start);
        
        obs_indices_resident[i] = index_ptr;
        obs_data_resident[i] = data_ptr;
        obs_counts_resident[i] = count;
    }
}

// dense to sparse conversion
sparse_data::sparse_data(const uint32_t _n_obs,
                         const uint32_t _n_samples,
                         const double* const *data)
  : n_obs(_n_obs)
  , n_samples(_n_samples)
  , clean_on_destruction(true)
  , obs_indices_resident(NULL)
  , obs_data_resident(NULL)
  , obs_counts_resident(NULL) {

    if (n_obs>0) {
        malloc_resident();
        unsigned int start = 0;
        for (uint32_t i = 0; i < n_obs; i++) {
            // pre-allocate with max size
            // the waste is stypically acceptable
            obs_data_resident[i]    = malloc_wcheck<double>(n_samples);
            obs_indices_resident[i] = malloc_wcheck<uint32_t>(n_samples);

            unsigned int cnt = 0;
            for (uint32_t j=0; j<n_samples; j++) {
              double val = data[j][i];
              if (val>0.0) {
                obs_data_resident[i][cnt] = val;
                obs_indices_resident[i][cnt] = j;
                cnt++;
              }
            }
            obs_counts_resident[i]  = cnt;
            start+=cnt;
        }
    }
}

sparse_data::~sparse_data() {
    if(clean_on_destruction) {
        if(obs_indices_resident != NULL && obs_data_resident != NULL) {
            for(unsigned int i = 0; i < n_obs; i++) {
                if(obs_indices_resident[i] != NULL)
                    free(obs_indices_resident[i]);
                if(obs_data_resident[i] != NULL)
                    free(obs_data_resident[i]);
            }
        }
    } 
    // else, it is the responsibility of the entity constructing this object
    // to clean itself up
    free_resident();
}

void sparse_data::malloc_resident() { 
    /* load obs sparse data */
    obs_indices_resident = malloc_wcheck<uint32_t*   >(n_obs);
    obs_data_resident    = malloc_wcheck<double*     >(n_obs);
    obs_counts_resident  = malloc_wcheck<unsigned int>(n_obs);
}

void sparse_data::free_resident() { 
    if(obs_indices_resident != NULL) {
        free(obs_indices_resident);
        obs_indices_resident=NULL;
    }
    if(obs_data_resident != NULL) {
        free(obs_data_resident);
        obs_data_resident = NULL;
    }
    if(obs_counts_resident != NULL) {
       free(obs_counts_resident);
        obs_counts_resident = NULL;
    }
}

uint32_t sparse_data::count_filtered_els(uint32_t idx, const double sample_counts[], const double min_sample_counts) const {
   const uint32_t  cnt_els = obs_counts_resident[idx];
   const uint32_t *indices = obs_indices_resident[idx];
   uint32_t cnt = 0;
   for (uint32_t i=0; i<cnt_els; i++) {
      const uint32_t el_idx = indices[i];
      if (sample_counts[el_idx]>=min_sample_counts) cnt++;
   }
   return cnt;
}

void sparse_data::describe_internals() const {
  printf("==== start sparse_data ====\n");
  printf("n_obs= %3d\tn_samples=%3d\n",n_obs,n_samples);
  if (obs_counts_resident!=NULL) {
    printf("obs_counts_resident\n");
    for (uint32_t i=0; i<n_obs; i++) {
      printf("\t%3d %3d\n",i,obs_counts_resident[i]);
    }
  }
  if (obs_data_resident!=NULL) {
    printf("obs_indices_resident & obs_data_resident\n");
    for (uint32_t i=0; i<n_obs; i++) {
       uint32_t cnt = obs_counts_resident[i];
       for (uint32_t j=0; j<cnt; j++) {
         printf("\t%3d %3d %3d %7.1f\n",i,j,obs_indices_resident[i][j],obs_data_resident[i][j]);
       }
    }
  }
  printf("====  end sparse_data  ====\n");
}

// ========================== biom_inmem  ===============================

biom_inmem::biom_inmem(bool _clean_on_destruction) 
  : biom_interface()
  , resident_obj(_clean_on_destruction)
  , sample_counts(NULL)
  , obs_id_index()
  , sample_id_index()
  , sample_ids()
  , obs_ids()
{}

biom_inmem::biom_inmem(const biom_inmem& other, bool _clean_on_destruction)
  : biom_interface(other)
  , resident_obj(other.resident_obj,_clean_on_destruction)
  , sample_counts(malloc_and_copy<double>(other.n_samples, other.sample_counts))
  , obs_id_index(other.obs_id_index)
  , sample_id_index(other.sample_id_index)
  , sample_ids(other.sample_ids)
  , obs_ids(other.obs_ids)
{}

biom_inmem::biom_inmem(const biom_inmem &other, const double min_sample_counts)
  : biom_interface(other)
  , resident_obj(other.resident_obj,other.sample_counts, min_sample_counts)
  , sample_counts(malloc_wcheck<double>(resident_obj.n_samples))
  , obs_id_index()
  , sample_id_index()
  , sample_ids()
  , obs_ids()
{
    if ((resident_obj.n_obs>0) && (resident_obj.n_samples>0)) {
     #pragma omp parallel for schedule(static)
     for(int i = 0; i < 2; i++) {
        if (i==0) {
          // filter out zero obs
          uint32_t obs_cnt = 0;
          for (uint32_t i=0; i<other.n_obs; i++) {
            if (resident_obj.obs_counts_resident[i]>0) { 
              obs_ids.push_back(other.obs_ids[i]); //keep
              if (i!=obs_cnt) {
                 // since we are increasing, this is safe
                 resident_obj.obs_counts_resident[obs_cnt]  = resident_obj.obs_counts_resident[i];
                 resident_obj.obs_data_resident[obs_cnt]    = resident_obj.obs_data_resident[i];
                 resident_obj.obs_indices_resident[obs_cnt] = resident_obj.obs_indices_resident[i];
              }
              obs_cnt++;
            }
          }
          n_obs = obs_cnt;
          resident_obj.n_obs = obs_cnt;
          // I could resize the buffers, but they are small enough to not be worth it
          create_id_index(obs_ids, obs_id_index);
        } else if(i == 1) {
          // resident_obj computed the proper n_samples during initialization
          n_samples = resident_obj.n_samples;
          sample_ids.reserve(n_samples);

          uint32_t i_my = 0;
          for (uint32_t i=0; i<other.n_samples; i++) {
             if (other.sample_counts[i]>=min_sample_counts) {
                sample_counts[i_my] = other.sample_counts[i];
                i_my++;
                sample_ids.push_back(other.sample_ids[i]);
             }
           }
           create_id_index(sample_ids, sample_id_index);
        }
     }
   } //((resident_obj.n_obs>0) && (resident_obj.n_samples>0))
   else
   {
     // degenerate case, just set to 0
     n_samples = 0;
     n_obs = 0;
   }
}


// not using const on indices/indptr/data as the pointers are being borrowed
biom_inmem::biom_inmem(const char* const * obs_ids_in,
                       const  char* const * samp_ids_in,
                       uint32_t* indices,
                       uint32_t* indptr,
                       double* data,
                       const int _n_obs,
                       const int _n_samples)
  : biom_interface(_n_samples, _n_obs)
  , resident_obj(_n_obs,_n_samples,indices,indptr,data)
  , sample_counts(NULL)
  , obs_id_index()
  , sample_id_index()
  , sample_ids()
  , obs_ids() {

    #pragma omp parallel for schedule(static)
    for(int x = 0; x < 2; x++) {
        if(x == 0) {
            obs_ids.resize(n_obs);
            for(uint32_t i = 0; i < n_obs; i++) {
                obs_ids[i] = std::string(obs_ids_in[i]);
            }
        } else {
            sample_ids.resize(n_samples);
            for(uint32_t i = 0; i < n_samples; i++) {
                sample_ids[i] = std::string(samp_ids_in[i]);
            }
        }
    }

    /* define a mapping between an ID and its corresponding offset */

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < 3; i++) {
        if(i == 0)
            create_id_index(obs_ids, obs_id_index);
        else if(i == 1)
            create_id_index(sample_ids, sample_id_index);
        else if(i == 2)
            compute_sample_counts();
    }

}

biom_inmem::biom_inmem(const char* const * obs_ids_in,
                       const char* const * samp_ids_in,
                       const double* const * data,
                       const int _n_obs,
                       const int _n_samples)
  : biom_interface(_n_samples, _n_obs)
  , resident_obj(_n_obs,_n_samples,data)
  , sample_counts(NULL)
  , obs_id_index()
  , sample_id_index()
  , sample_ids()
  , obs_ids() {

    #pragma omp parallel for schedule(static)
    for(int x = 0; x < 2; x++) {
        if(x == 0) {
            obs_ids.resize(n_obs);
            for(uint32_t i = 0; i < n_obs; i++) {
                obs_ids[i] = std::string(obs_ids_in[i]);
            }
        } else {
            sample_ids.resize(n_samples);
            for(uint32_t i = 0; i < n_samples; i++) {
                sample_ids[i] = std::string(samp_ids_in[i]);
            }
        }
    }

    /* define a mapping between an ID and its corresponding offset */

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < 3; i++) {
        if(i == 0)
            create_id_index(obs_ids, obs_id_index);
        else if(i == 1)
            create_id_index(sample_ids, sample_id_index);
        else if(i == 2)
            compute_sample_counts();
    }

}

biom_inmem::~biom_inmem() {
    if (sample_counts!=NULL) free(sample_counts);
}

void biom_inmem::create_id_index(const std::vector<std::string> &ids, 
                                 std::unordered_map<std::string, uint32_t> &map) {
    uint32_t count = 0;
    map.reserve(ids.size());
    for(auto i = ids.begin(); i != ids.end(); i++, count++) {
        map[*i] = count;
    }
}

template<class TFloat>
void biom_inmem::get_obs_data_TT(const uint32_t idx, TFloat* out) const {
    unsigned int count = resident_obj.obs_counts_resident[idx];
    const uint32_t * const indices = resident_obj.obs_indices_resident[idx];
    const double * const data = resident_obj.obs_data_resident[idx];

    // reset our output buffer
    for(unsigned int i = 0; i < n_samples; i++)
        out[i] = 0.0;
    
    for(unsigned int i = 0; i < count; i++) {
        out[indices[i]] = data[i];
    }
}

template<class TFloat>
void biom_inmem::get_obs_data_TT(const std::string &id, TFloat* out) const {
    uint32_t idx = obs_id_index.at(id);
    get_obs_data_TT<TFloat>(idx,out);
}

void biom_inmem::get_obs_data(const std::string &id, double* out) const {
  biom_inmem::get_obs_data_TT(id,out);
}

void biom_inmem::get_obs_data(const std::string &id, float* out) const {
  biom_inmem::get_obs_data_TT(id,out);
}


// note: out is supposed to be fully filled, i.e. out[start:end]
template<class TFloat>
void biom_inmem::get_obs_data_range_TT(const uint32_t idx, unsigned int start, unsigned int end, bool normalize, TFloat* out) const {
    unsigned int count = resident_obj.obs_counts_resident[idx];
    const uint32_t * const indices = resident_obj.obs_indices_resident[idx];
    const double * const data = resident_obj.obs_data_resident[idx];

    // reset our output buffer
    for(unsigned int i = start; i < end; i++)
        out[i-start] = 0.0;

    if (normalize) {
      for(unsigned int i = 0; i < count; i++) {
        const uint32_t j = indices[i];
        if ((j>=start)&&(j<end)) { 
          out[j-start] = data[i]/sample_counts[j];
        }
      }
    } else {
      for(unsigned int i = 0; i < count; i++) {
        const uint32_t j = indices[i];
        if ((j>=start)&&(j<end)) {
          out[j-start] = data[i];
        }
      }
    }
}

template<class TFloat>
void biom_inmem::get_obs_data_range_TT(const std::string &id, unsigned int start, unsigned int end, bool normalize, TFloat* out) const {
    uint32_t idx = obs_id_index.at(id);
    get_obs_data_range_TT<TFloat>(idx,start,end,normalize,out);
}

void biom_inmem::get_obs_data_range(const std::string &id, unsigned int start, unsigned int end, bool normalize, double* out) const {
  biom_inmem::get_obs_data_range_TT(id,start,end,normalize,out);
}

void biom_inmem::get_obs_data_range(const std::string &id, unsigned int start, unsigned int end, bool normalize, float* out) const {
  biom_inmem::get_obs_data_range_TT(id,start,end,normalize,out);
}

void biom_inmem::compute_sample_counts() {
    sample_counts = (double*)calloc(sizeof(double), n_samples);

    for(unsigned int i = 0; i < n_obs; i++) {
        unsigned int count = resident_obj.obs_counts_resident[i];
        uint32_t *indices = resident_obj.obs_indices_resident[i];
        double *data = resident_obj.obs_data_resident[i];
        for(unsigned int j = 0; j < count; j++) {
            uint32_t index = indices[j];
            double datum = data[j];
            sample_counts[index] += datum;
        }
    }
}

const double *biom_inmem::get_sample_counts() const {
  return sample_counts;
}
const std::vector<std::string> &biom_inmem::get_sample_ids() const {
  return sample_ids;
}

const std::vector<std::string> &biom_inmem::get_obs_ids() const {
  return obs_ids;
}

void biom_inmem::describe_internals() const {
  resident_obj.describe_internals();
  printf("==== start biom_inmem ====\n");
  printf("n_obs= %3d\tn_samples=%3d\n",n_obs,n_samples);
  if (sample_counts!=NULL) {
    printf("sample_counts\n");
    for (uint32_t i=0; i<n_samples; i++) {
      printf("\t%3d %7.1f\n",i,sample_counts[i]);
    }
  }
  printf("sample_ids\n");
  for (uint32_t i=0; i<n_samples; i++) {
    printf("\t%3d %s\n",i,sample_ids.at(i).c_str());
  }
  printf("obs_ids\n");
  for (uint32_t i=0; i<n_obs; i++) {
    printf("\t%3d %s\n",i,obs_ids.at(i).c_str());
  }
  
  printf("====   nd biom_inmem  ====\n");
}
