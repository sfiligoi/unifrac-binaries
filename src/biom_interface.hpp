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
            uint32_t n_samples;  // the number of samples
            uint32_t n_obs;      // the number of observations

            /* default constructor */
            biom_interface() 
             : n_samples(0), n_obs(0) {}

            /* full constructor */
            biom_interface(uint32_t _n_samples, uint32_t _n_obs) 
             : n_samples(_n_samples), n_obs(_n_obs) {}

            /* copy constructor */
            biom_interface(const biom_interface& other) 
             : n_samples(other.n_samples), n_obs(other.n_obs) {}

            /* default destructor
             *
             * Need a virtual one to allow for polymorphism
             */
            virtual ~biom_interface() {}

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

            // cache the IDs contained within the table
            virtual const std::vector<std::string> &get_sample_ids() const =0;
            virtual const std::vector<std::string> &get_obs_ids() const = 0;

            /* get the pre-comoputed counts */
            virtual const double *get_sample_counts() const =0;
    };
}

#endif /* _UNIFRAC_BIOOM_INTERFACE_H */
