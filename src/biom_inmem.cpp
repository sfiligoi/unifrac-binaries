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
        for(unsigned int i = 0; i < n_obs; i++) {
            unsigned int cnt = other.obs_counts_resident[i];
            obs_counts_resident[i] = cnt;
            obs_data_resident[i] = copy_resident_el<double>(cnt, other.obs_data_resident[i]);
            obs_indices_resident[i] = copy_resident_el<uint32_t>(cnt, other.obs_indices_resident[i]);
        }
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
    obs_indices_resident = (uint32_t**)malloc(sizeof(uint32_t*) * n_obs);
    if(obs_indices_resident == NULL) {
        fprintf(stderr, "Failed to allocate %zd bytes; [%s]:%d\n", 
                sizeof(uint32_t**) * n_obs, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    obs_data_resident = (double**)malloc(sizeof(double*) * n_obs);
    if(obs_data_resident == NULL) {
        fprintf(stderr, "Failed to allocate %zd bytes; [%s]:%d\n", 
                sizeof(double**) * n_obs, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    obs_counts_resident = (unsigned int*)malloc(sizeof(unsigned int) * n_obs);
    if(obs_counts_resident == NULL) {
        fprintf(stderr, "Failed to allocate %zd bytes; [%s]:%d\n", 
                sizeof(unsigned int) * n_obs, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
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
template<class TData>
TData *sparse_data::copy_resident_el(unsigned int cnt, const TData *other) const { 
    unsigned int bufsize = sizeof(TData) * cnt;
    TData *my = (TData *)malloc(bufsize);
    if(my == NULL) {
        fprintf(stderr, "Failed to allocate %d bytes; [%s]:%d\n", 
                bufsize, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
   std::memcpy(my, other, bufsize);
   return my;
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
  , sample_counts(NULL)
  , obs_id_index(other.obs_id_index)
  , sample_id_index(other.sample_id_index)
  , sample_ids(other.sample_ids)
  , obs_ids(other.obs_ids)
{
    // we re-create this every time
    compute_sample_counts();
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
            for(int i = 0; i < n_obs; i++) {
                obs_ids[i] = std::string(obs_ids_in[i]);
            }
        } else {
            sample_ids.resize(n_samples);
            for(int i = 0; i < n_samples; i++) {
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
        const int32_t j = indices[i];
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

