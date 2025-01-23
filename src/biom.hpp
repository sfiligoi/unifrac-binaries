/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2016-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */


#ifndef _UNIFRAC_BIOM_H
#define _UNIFRAC_BIOM_H

#include <H5Cpp.h>
#include <H5Dpublic.h>

#include "biom_inmem.hpp"

namespace su {
    class biom : public biom_inmem {
        public:
            /* nullary constructor */
            biom();

            /* default constructor
             *
             * @param filename The path to the BIOM table to read
             */
            biom(std::string filename);

            /* constructor from compress sparse data
             * Note: deprecated, use biom_inmem directly, instead 
             *
             * @param obs_ids vector of observation identifiers
             * @param samp_ids vector of sample identifiers
             * @param index vector of index positions
             * @param indptr vector of indptr positions
             * @param data vector of observation counts
             * @param n_obs number of observations
             * @param n_samples number of samples
             * @param nnz number of data points
             */
            biom(const char* const * obs_ids,
                 const char* const * samp_ids,
                 uint32_t* index,
                 uint32_t* indptr,
                 double* data,
                 const int n_obs,
                 const int n_samples,
                 const int nnz);

            /* default destructor
             *
             * Temporary arrays are freed
             */
            virtual ~biom();

            /* prevent default copy contructors and operators from being generated */
            biom(const biom& other) = delete;
            biom& operator= (const biom&) = delete;

        private:
            /* datasets defined by the BIOM 2.x spec */ 
            static constexpr const char * OBS_INDPTR  = "/observation/matrix/indptr";
            static constexpr const char * OBS_INDICES = "/observation/matrix/indices";
            static constexpr const char * OBS_DATA    = "/observation/matrix/data";
            static constexpr const char * OBS_IDS     = "/observation/ids";

            static constexpr const char * SAMPLE_INDPTR  = "/sample/matrix/indptr";
            static constexpr const char * SAMPLE_INDICES = "/sample/matrix/indices";
            static constexpr const char * SAMPLE_DATA    = "/sample/matrix/data";
            static constexpr const char * SAMPLE_IDS     = "/sample/ids";

            bool has_hdf5_backing = false;
            
            // cache both index pointers into both CSC and CSR representations
            std::vector<uint32_t> sample_indptr;
            std::vector<uint32_t> obs_indptr;

            /* retain DataSet handles within the HDF5 file */
            H5::DataSet obs_indices;
            H5::DataSet sample_indices;
            H5::DataSet obs_data;
            H5::DataSet sample_data;
            H5::H5File file;
            
            unsigned int get_obs_data_direct(const std::string &id, uint32_t *& current_indices_out, double *& current_data_out);
            unsigned int get_sample_data_direct(const std::string &id, uint32_t *& current_indices_out, double *& current_data_out);

            /* load ids from an axis
             *
             * @param path The dataset path to the ID dataset to load
             * @param ids The variable representing the IDs to load into
             */          
            void load_ids(const char *path, std::vector<std::string> &ids);

            /* load the index pointer for an axis
             *
             * @param path The dataset path to the index pointer to load
             * @param indptr The vector to load the data into
             */
            void load_indptr(const char *path, std::vector<uint32_t> &indptr);

            /* count the number of nonzero values and set nnz */
            void set_nnz();
        public:
            uint32_t nnz;        // the total number of nonzero entries

	    // used by ssu for helper messages
            static inline size_t load_n_samples(const char* filename) {
                const char* path = SAMPLE_IDS;
		H5::H5File file(filename, H5F_ACC_RDONLY);
		H5::DataSet ds_ids = file.openDataSet(path);
		H5::DataSpace dataspace = ds_ids.getSpace();

                hsize_t dims[1];
                dataspace.getSimpleExtentDims(dims, NULL);

		return dims[0];
            }

            // for unit testing
            bool is_sample_indptr(const std::vector<uint32_t>& other) const { return sample_indptr==other; }
            bool is_obs_indptr(const std::vector<uint32_t>& other) const { return obs_indptr==other; }

    };
}

#endif /* _UNIFRAC_BIOM_H */

