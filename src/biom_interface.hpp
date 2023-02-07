/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */


#ifndef _UNIFRAC_BIOM_INTERFACE_H
#define _UNIFRAC_BIOM_INTERFACE_H

#include <vector>
#include <string>

namespace su {
    class biom_interface {
        public:
            // cache the IDs contained within the table
            std::vector<std::string> sample_ids;
            std::vector<std::string> obs_ids;

            uint32_t n_samples;  // the number of samples
            uint32_t n_obs;      // the number of observations
            uint32_t nnz;        // the total number of nonzero entries
            double *sample_counts;

            /* default constructor
             *
             * Automatically create the needed objects.
             * All other initialization happens in children constructors.
             */
            biom_interface() 
             : sample_ids(), obs_ids()
             , n_samples(0), n_obs(0), nnz(0)
             , sample_counts(NULL) {}

            /* default destructor
             *
             * Automatically destroy the objects.
             * All other cleanup must have been performed by the children constructors.
             */
            virtual ~biom_interface() {
               if (sample_counts!=NULL) free(sample_counts);
            }

            /* get a dense vector of observation data
             *
             * @param id The observation ID to fetch
             * @param out An allocated array of at least size n_samples. 
             *      Values of an index position [0, n_samples) which do not
             *      have data will be zero'd.
             */
            virtual void get_obs_data(const std::string &id, double* out) const = 0; 
            virtual void get_obs_data(const std::string &id, float* out) const = 0;

            /* get a dense vector of a range of observation data
             *
             * @param id The observation ID to fetc
             * @param start Initial index
             * @param end   First index past the end
             * @param normalize If set, divide by sample_counts
             * @param out An allocated array of at least size (end-start). First element will corrrectpoint to index start. 
             *      Values of an index position [0, (end-start)) which do not
             *      have data will be zero'd.
             */
            virtual void get_obs_data_range(const std::string &id, unsigned int start, unsigned int end, bool normalize, double* out) const = 0;
            virtual void get_obs_data_range(const std::string &id, unsigned int start, unsigned int end, bool normalize, float* out) const = 0;
    };
}

#endif /* _UNIFRAC_BIOOM_INTERFACE_H */
