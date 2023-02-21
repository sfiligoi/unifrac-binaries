/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2023-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <random>

#include "biom_subsampled.hpp"

using namespace su;

linked_sparse_transposed::linked_sparse_transposed(sparse_data &other) 
  : n_obs(other.n_samples)
  , n_samples(other.n_obs) {
    obs_counts_resident = (unsigned int*)calloc(sizeof(unsigned int), n_obs);
   if(obs_counts_resident == NULL) {
        fprintf(stderr, "Failed to allocate %zd bytes; [%s]:%d\n", 
                sizeof(unsigned int) * n_obs, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
   }

   for (uint32_t i=0; i<other.n_obs; i++) {
     const uint32_t cnt = other.obs_counts_resident[i];
     const uint32_t * idxs = other.obs_indices_resident[i];
     for (uint32_t j=0; j<cnt; j++) {
       obs_counts_resident[idxs[j]]++;
     }
   }

   obs_data_resident = (double***)malloc(sizeof(double**) * n_obs);
   if(obs_data_resident == NULL) {
        fprintf(stderr, "Failed to allocate %zd bytes; [%s]:%d\n", 
                sizeof(double**) * n_obs, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
   }
   uint32_t _max_count = 0;
   for (uint32_t i=0; i<n_obs; i++) {
     const uint32_t cnt = obs_counts_resident[i];
     _max_count = std::max(_max_count,cnt);
     obs_data_resident[i] = (double**)malloc(sizeof(double*) * cnt);
     if(obs_data_resident[i] == NULL) {
        fprintf(stderr, "Failed to allocate %zd bytes; [%s]:%d\n", 
                sizeof(double*) * cnt, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
     }
   }
   max_count = _max_count;

   for (uint32_t i=0; i<other.n_obs; i++) {
     const uint32_t cnt = other.obs_counts_resident[i];
     const uint32_t *idxs = other.obs_indices_resident[i];
     double *data = other.obs_data_resident[i];
     for (uint32_t j=0; j<cnt; j++) {
       uint32_t local_i = idxs[j];
       obs_data_resident[local_i][0] = &(data[j]);
       obs_data_resident[local_i]++;
     }
   }

   // we moved the pointers in place, rewind
   for (uint32_t i=0; i<n_obs; i++) {
      const uint32_t cnt = obs_counts_resident[i];
      obs_data_resident[i] -=  cnt;
   }
}

linked_sparse_transposed::~linked_sparse_transposed() {
    if(obs_data_resident != NULL) {
        free(obs_data_resident);
        obs_data_resident = NULL;
    }
    if(obs_counts_resident != NULL) {
       free(obs_counts_resident);
        obs_counts_resident = NULL;
    }
}

void linked_sparse_transposed::transposed_subsample_with_replacement(const uint32_t n, const uint32_t random_seed) {
    std::default_random_engine generator(random_seed);

    // use common buffer to minimize allocation costs
    double *data_in = new double[max_count];  // original values
    uint32_t *data_out = new uint32_t[max_count]; // computed values

    for (uint32_t i=0; i<n_obs; i++) {
        unsigned int length = obs_counts_resident[i];
        double* *data_arr = obs_data_resident[i];

        for (unsigned int j=0; j<length; j++) data_in[j] = *(data_arr[j]);
        
        std::discrete_distribution<uint32_t> multinomial(data_in, data_in+length);
        for (unsigned int j=0; j<length; j++) data_out[j] = 0;
        for (uint32_t j=0; j<n; j++) data_out[multinomial(generator)]++;

        for (unsigned int j=0; j<length; j++) *(data_arr[j]) = data_out[j];
    }
    delete[] data_out;
    delete[] data_in;
}

// =====================  sparse_data_subsampled  ==========================

void sparse_data_subsampled::subsample_with_replacement(const uint32_t n, const uint32_t random_seed) {
    linked_sparse_transposed transposed(*this);
    transposed.transposed_subsample_with_replacement(n,random_seed);
}

// =====================  biom_subsampled  ==========================

biom_subsampled::biom_subsampled(const biom_inmem &parent, const uint32_t n, const uint32_t random_seed) 
  : biom_inmem(true)
{
   sparse_data_subsampled tmp_obj(parent.get_resident_obj(), true);
   tmp_obj.subsample_with_replacement(n,random_seed);
   steal_nonzero(parent,tmp_obj);

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

void biom_subsampled::steal_nonzero(const biom_inmem &parent, sparse_data& subsampled_obj) {
   // initialize data structures
   resident_obj.n_obs = parent.n_obs;
   resident_obj.n_samples = parent.n_samples;
   resident_obj.malloc_resident();
   obs_ids.reserve(parent.n_obs);

   const std::vector<std::string> &parent_obs_ids = parent.get_obs_ids();

   // now do the copy
   n_obs = 0;
   for (uint32_t i=0; i<parent.n_obs; i++) {
     const uint32_t cnt = subsampled_obj.obs_counts_resident[i];
     double *data = subsampled_obj.obs_data_resident[i];
     uint32_t nz = 0; 
     for (uint32_t j=0; j<cnt; j++) if (data[j]>0.0) nz++;

     if (nz>0) {
        // steal non-zero data
        resident_obj.obs_indices_resident[n_obs] = subsampled_obj.steal_indices(i);
        resident_obj.obs_data_resident[n_obs] = subsampled_obj.steal_data(i); 
        resident_obj.obs_counts_resident[n_obs] = cnt;
        obs_ids.push_back(parent_obs_ids[i]);
        n_obs++;
     }
     // else just ignore
   }

   sample_ids = parent.get_sample_ids();;
   resident_obj.n_obs = n_obs;
   n_samples = parent.n_samples;
}


