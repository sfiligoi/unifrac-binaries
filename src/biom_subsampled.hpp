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

        public:  // keep it open for ease of access
            uint32_t n_obs;     // row dimension
            uint32_t n_samples; // column dimension
            
            double* **obs_data_resident;
            unsigned int *obs_counts_resident;
            uint32_t max_count; // max(obs_counts_resident[])

    };

    class biom_subsampled : public biom_inmem {
        public:
            /* default constructor
             *
             * @param parent biom object to subsample
             * @param n Number of items to subsample
             */
            biom_subsampled(const biom_inmem &parent, const uint32_t n);

            /* default destructor */
            virtual ~biom_subsampled();

        protected:

            /* perform subsampling with replacement, no filtering */
            void init_with_replacement(const uint32_t n);
        public:
            /* prevent default copy contructor and operator from being generated */
            biom_subsampled(const biom_subsampled& other) = delete;
            biom_subsampled& operator= (const biom_subsampled&) = delete;
    };
}

#endif /* _UNIFRAC_SUBSAMPLED_H */

