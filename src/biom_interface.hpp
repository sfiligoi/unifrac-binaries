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

            /* default constructor
             *
             * Automatically create the needed objects.
             * All other initialization happens in children constructors.
             */
            biom_interface() 
             : sample_ids(), obs_ids()
             , n_samples(0), n_obs(0), nnz(0) {}

            /* default destructor
             *
             * Need a virtual one to allow for polymorphism
             */
            virtual ~biom_interface() {}

            /* prevent default copy contructors and operators from being generated */
            biom_interface(const biom_interface& other) = delete;
            biom_interface& operator= (const biom_interface&) = delete;

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

            /* get the pre-comoputed counts */
             virtual const double *get_sample_counts() const =0;
    };
}

#endif /* _UNIFRAC_BIOOM_INTERFACE_H */
