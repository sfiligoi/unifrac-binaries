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

        public:
            /* prevent default copy contructor and operator from being generated */
            biom_subsampled(const biom_subsampled& other) = delete;
            biom_subsampled& operator= (const biom_subsampled&) = delete;
    };
}

#endif /* _UNIFRAC_SUBSAMPLED_H */

