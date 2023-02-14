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
   for (uint32_t i=0; i<n_obs; i++) {
     const uint32_t cnt = obs_counts_resident[i];
     obs_data_resident[i] = (double**)malloc(sizeof(double*) * cnt);
     if(obs_data_resident[i] == NULL) {
        fprintf(stderr, "Failed to allocate %zd bytes; [%s]:%d\n", 
                sizeof(double*) * cnt, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
     }
   }

   uint32_t _max_count = 0;
   for (uint32_t i=0; i<other.n_obs; i++) {
     const uint32_t cnt = other.obs_counts_resident[i];
     const uint32_t *idxs = other.obs_indices_resident[i];
     double *data = other.obs_data_resident[i];
     _max_count = std::max(_max_count,cnt);
     for (uint32_t j=0; j<cnt; j++) {
       uint32_t local_i = idxs[j];
       obs_data_resident[local_i][0] = &(data[j]);
       obs_data_resident[local_i]++;
     }
   }
   max_count = _max_count;

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

biom_subsampled::biom_subsampled(const biom_inmem &parent, const uint32_t n) 
  : biom_inmem(parent, true)
{
   init_with_replacement(n);
}

biom_subsampled::~biom_subsampled()
{}

void biom_subsampled::init_with_replacement(const uint32_t n) {
    linked_sparse_transposed transposed(this->resident_obj);

    // TODO: This is just a temporary hack
    // construct a trivial random generator engine
    std::default_random_engine generator(0);

    // use common buffer to minimize allocation costs
    double *data_in = new double[transposed.max_count];  // original values
    uint32_t *data_out = new uint32_t[transposed.max_count]; // computed values

    for (uint32_t i=0; i<n_obs; i++) {
        unsigned int length = transposed.obs_counts_resident[i];
        double* *data_arr = transposed.obs_data_resident[i];

        for (unsigned int j=0; j<length; j++) data_in[j] = *(data_arr[j]);
        
        std::discrete_distribution<uint32_t> multinomial(data_in, data_in+length);
        for (unsigned int j=0; j<length; j++) data_out[j] = 0;
        for (uint32_t j=0; j<n; j++) data_out[multinomial(generator)]++;

        for (unsigned int j=0; j<length; j++) *(data_arr[j]) = data_out[j];
    }
    delete[] data_out;
    delete[] data_in;
}

