/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2016-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */


#ifndef _UNIFRAC_BIOM_INMEM_H
#define _UNIFRAC_BIOM_INMEM_H

#include <vector>
#include <unordered_map>

#include "biom_interface.hpp"

/* C version of the biom_inmem data
 *
 * obs_ids <char**> the observation IDs
 * sample_ids <char**> the sample IDs
 * indices <int32_t**> the indices of the data values
 * data <double**> the actual matrix values
 * counts <int32_t*> raw counts
 * sample_counts <double*> distilled counts
 * n_obs <int> the number of observations, corresponding to length of obs_ids
 * n_samples <int> the number of samples, corresponding to the length of sample_ids
 */
typedef struct su_c_biom_inmem {
    char** obs_ids;
    char** sample_ids;
    uint32_t **indices;
    double **data;
    uint32_t *counts;
    double *sample_counts;
    int n_obs;
    int n_samples;
} su_c_biom_inmem_t;

/* C version of the biom_inmem data in a sparse format
 *
 * obs_ids <char**> the observation IDs
 * sample_ids <char**> the sample IDs
 * indices <int32_t*> the indices of the data values
 * indptr <int32_t*> the row offset of the data values
 * data <double*> the actual matrix values
 * n_obs <int> the number of observations, corresponding to length of obs_ids
 * n_samples <int> the number of samples, corresponding to the length of sample_ids
 */
typedef struct su_c_biom_sparse {
    char** obs_ids;
    char** sample_ids;
    uint32_t* indices;
    uint32_t* indptr;
    double* data;
    int n_obs;
    int n_samples;
} su_c_biom_sparse_t;

namespace su {
    class sparse_data {
        public:
            /* default constructor */
            sparse_data(bool _clean_on_destruction);

            /* constructor from C structure
             *
             * @param n_obs number of observations
             * @param indices array of vectors of index positions
             * @param data array of vectors of observation counts
             * @param counts vector of counts
             */
            sparse_data(const uint32_t n_obs,
                        const uint32_t n_samples,
                        uint32_t** indices,
                        double** data,
                        uint32_t *counts);

            /* constructor from compress sparse data
             *
             * @param n_obs number of observations
             * @param index vector of index positions
             * @param indptr vector of indptr positions
             * @param data vector of observation counts
             */
            sparse_data(const uint32_t n_obs,
                        const uint32_t n_samples,
                        uint32_t* index,
                        uint32_t* indptr,
                        double* data);

            /* default destructor */
            virtual ~sparse_data();

            /* modified copy constructor */
            sparse_data(const sparse_data& other, bool _clean_on_destruction);

            /* modified copy constructor
             *
             * @param other             Object to copy
             * @param sample_counts     Counts associarted with the object
             * @param min_sample_counts Minimum number of counts needed to keep a sample
             */
            sparse_data(const sparse_data& other, const double sample_counts[], const double min_sample_counts);

            /* remove ownership of this obs_idx buffer */
            uint32_t *steal_indices(uint32_t obs_idx) {uint32_t *out=obs_indices_resident[obs_idx]; obs_indices_resident[obs_idx]=NULL; return out;}
            double *steal_data(uint32_t obs_idx) {double *out=obs_data_resident[obs_idx]; obs_data_resident[obs_idx]=NULL; return out;}

            /* prevent default copy constructor and operator from being generated */
            sparse_data(const sparse_data& other) = delete;
            sparse_data& operator= (const sparse_data&) = delete;

            /* Helper functions */
            void malloc_resident();
            void free_resident();
            uint32_t count_filtered_els(uint32_t idx, const double sample_counts[], const double min_sample_counts) const;

        public:  // keep it open for ease of access
            uint32_t n_obs;     // row dimension
            uint32_t n_samples; // column dimension
            bool clean_on_destruction;
            
            uint32_t **obs_indices_resident;
            double **obs_data_resident;
            uint32_t *obs_counts_resident;

            // debug helper functions
            void describe_internals() const;
    };

    class biom_inmem : public biom_interface {
        public:
            /* default constructor */
            biom_inmem(bool _clean_on_destruction);

            /* contructor from the C structure
             * Note: Not using const as we will link to the existing structures
             *       to avoid copying big buffers.
             */
            biom_inmem(su_c_biom_inmem_t &other);

            /* contructor from the sparse C structure
             * Note: Not using const as we will link to the existing structures
             *       to avoid copying big buffers.
             */
            biom_inmem(su_c_biom_sparse_t &other);

            /* filtering constructor
             *
             * @param other biom object to filter
             * @param min_sample_counts Minimum number of counts needed to keep a sample
             */
            biom_inmem(const biom_inmem &other, const double min_sample_counts);

            /* Modified copy constructor */
            biom_inmem(const biom_inmem& other, bool _clean_on_destruction);

            /* default destructor */
            virtual ~biom_inmem();

            /* get a dense vector of observation data
             *
             * @param id The observation ID to fetch
             * @param out An allocated array of at least size n_samples. 
             *      Values of an index position [0, n_samples) which do not
             *      have data will be zero'd.
             */
            void get_obs_data(const std::string &id, double* out) const; 
            void get_obs_data(const std::string &id, float* out) const;

            /* get a dense vector of a range of observation data
             *
             * @param id The observation ID to fetch
             * @param start Initial index
             * @param end   First index past the end
             * @param normalize If set, divide by sample_counts
             * @param out An allocated array of at least size (end-start). First element will corrrectpoint to index start. 
             *      Values of an index position [0, (end-start)) which do not
             *      have data will be zero'd.
             */
            void get_obs_data_range(const std::string &id, unsigned int start, unsigned int end, bool normalize, double* out) const;
            void get_obs_data_range(const std::string &id, unsigned int start, unsigned int end, bool normalize, float* out) const;

            /* getters to local variables */
            virtual const std::vector<std::string> &get_sample_ids() const;
            virtual const std::vector<std::string> &get_obs_ids() const;
            virtual const double *get_sample_counts() const;

            /* get the C version of the object 
             *
             * Note: The pointers link to internal structures in this object
             *       so this object must outlive the C equivalent.
             */
            void get_c_struct(su_c_biom_inmem_t& c_data);
        protected:
            sparse_data resident_obj;

            // distilled from the above resident values
            double *sample_counts;

            /* At construction, lookups mapping IDs -> index position within an
             * axis are defined
             */
            std::unordered_map<std::string, uint32_t> obs_id_index;
            std::unordered_map<std::string, uint32_t> sample_id_index;
 
            // cache the IDs contained within the table
            std::vector<std::string> sample_ids;
            std::vector<std::string> obs_ids;

            // optional C equivalents of sample_ids and obs_ids
            // used for persistency
            char** c_obs_ids;
            char** c_sample_ids;

        protected:

            void compute_sample_counts();
            void build_c_ids();


            /* create an index mapping an ID to its corresponding index 
             * position.
             *
             * @param ids A vector of IDs to index
             * @param map A hash table to populate
             */
            void create_id_index(const std::vector<std::string> &ids, 
                                 std::unordered_map<std::string, uint32_t> &map);


            // templatized version
            template<class TFloat> void get_obs_data_TT(const uint32_t idx, TFloat* out) const;
            template<class TFloat> void get_obs_data_TT(const std::string &id, TFloat* out) const;
            template<class TFloat> void get_obs_data_range_TT(const uint32_t idx, unsigned int start, unsigned int end, bool normalize, TFloat* out) const;
            template<class TFloat> void get_obs_data_range_TT(const std::string &id, unsigned int start, unsigned int end, bool normalize, TFloat* out) const;
        public:
            const sparse_data& get_resident_obj() const {return resident_obj;}

            /* prevent default copy constructor and operator from being generated */
            biom_inmem(const biom_inmem& other) = delete;
            biom_inmem& operator= (const biom_inmem&) = delete;

            // debug helper functions
            void describe_internals() const;
    };

}

#endif /* _UNIFRAC_BIOM_H */

