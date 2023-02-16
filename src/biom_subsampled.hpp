/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2023-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */


#ifndef _UNIFRAC_BIOM_SUBSAMPLED_H
#define _UNIFRAC_BIOM_SUBSAMPLED_H

#include "biom_inmem.hpp"

namespace su {
    // Keep a transposed version of a sparse object
    class linked_sparse_transposed {
        public:
            /* default constructor */
            linked_sparse_transposed(sparse_data& other);

            /* default destructor */
            virtual ~linked_sparse_transposed();

            /* prevent default copy constructor and operator from being generated */
            linked_sparse_transposed(const linked_sparse_transposed& other) = delete;
            linked_sparse_transposed& operator= (const linked_sparse_transposed&) = delete;

            /* perform subsampling with replacement, no filtering */
            void transposed_subsample_with_replacement(const uint32_t n, const uint32_t random_seed);
        public:  // keep it open for ease of access
            uint32_t n_obs;     // row dimension
            uint32_t n_samples; // column dimension
            
            double* **obs_data_resident;
            unsigned int *obs_counts_resident;
            uint32_t max_count; // max(obs_counts_resident[])

    };

    class sparse_data_subsampled: public sparse_data {
        public:
            /* default constructor */
            sparse_data_subsampled(bool _clean_on_destruction)
             :sparse_data(_clean_on_destruction) {}

            /* modified copy constructor */
            sparse_data_subsampled(const sparse_data& other, bool _clean_on_destruction)
             : sparse_data(other,_clean_on_destruction) {}

            /* prevent default copy constructor and operator from being generated */
            sparse_data_subsampled(const sparse_data_subsampled& other) = delete;
            sparse_data_subsampled& operator= (const sparse_data_subsampled&) = delete;

            /* perform subsampling with replacement, no filtering */
            void subsample_with_replacement(const uint32_t n, const uint32_t random_seed);
    };

    class biom_subsampled : public biom_inmem {
        public:
            /* default constructor
             *
             * @param parent biom object to subsample
             * @param n Number of items to subsample
             */
            biom_subsampled(const biom_inmem &parent, const uint32_t n, const uint32_t random_seed);

        protected:
            void copy_nonzero(const biom_inmem &parent, sparse_data& subsampled_obj);

        public:
            /* prevent default copy contructor and operator from being generated */
            biom_subsampled(const biom_subsampled& other) = delete;
            biom_subsampled& operator= (const biom_subsampled&) = delete;
    };
}

#endif /* _UNIFRAC_SUBSAMPLED_H */

