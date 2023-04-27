/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2023-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

#ifndef __UNIFRAC_API_NV
#define __UNIFRAC_API_NV 1

#ifndef PARTIAL_MAGIC_V2
#include "api.hpp"
#endif

#include "biom_inmem.hpp"
#include "tree.hpp"

#ifdef __cplusplus
#define EXTERN extern "C"
#endif

/*
 * The following functions are equivalent to those in api.hpp
 * but are compiled with NVIDIA GPU support
 */

/* Check if we should use the NVIDIA version of the routines
 *
 * NVIDIA version is typically only useful when a GPU is detected.
 *
 */
EXTERN bool ssu_should_use_nv();

/* Compute UniFrac - condensed form
 *
 * biom_data <su_c_biom_inmem_t *> BIOM C data struct
 * tree_data <su_c_bptree_t *> BPTree C data struct
 * unifrac_method <const char*> the requested unifrac method.
 * variance_adjust <bool> whether to apply variance adjustment.
 * alpha <double> GUniFrac alpha, only relevant if method == generalized.
 * bypass_tips <bool> disregard tips, reduces compute by about 50%
 * n_substeps <uint> the number of substeps to use.
 * result <mat_t**> the resulting distance matrix in condensed form, this is initialized within the method so using **
 *
 * one_off returns the following error codes:
 *
 * okay           : no problems encountered
 * unknown_method : the requested method is unknown.
 * table_empty    : the table does not have any entries
 */
EXTERN ComputeStatus one_off_inmem_nv_fp64(su_c_biom_inmem_t *biom_data, su_c_bptree_t *tree_data,
                                           const char* unifrac_method, bool variance_adjust, double alpha,
                                           bool bypass_tips, unsigned int n_substeps, mat_t** result);


/* Compute a subset of a UniFrac distance matrix
 *
 * biom_data <su_c_biom_inmem_t *> BIOM C data struct
 * tree_data <su_c_bptree_t *> BPTree C data struct
 * unifrac_method <const char*> the requested unifrac method.
 * variance_adjust <bool> whether to apply variance adjustment.
 * alpha <double> GUniFrac alpha, only relevant if method == generalized.
 * bypass_tips <bool> disregard tips, reduces compute by about 50%
 * n_substeps <uint> the number of substeps to use.
 * stripe_start <uint> the starting stripe to compute
 * stripe_stop <uint> the last stripe to compute
 * result <partial_mat_t**> the resulting distance matrix in condensed form, this is initialized within the method so using **
 *
 * partial returns the following error codes:
 *
 * okay           : no problems encountered
 * table_missing  : the filename for the table does not exist
 * tree_missing   : the filename for the tree does not exist
 * unknown_method : the requested method is unknown.
 */

EXTERN ComputeStatus partial_inmem_nv(su_c_biom_inmem_t *biom_data, su_c_bptree_t *tree_data,
                                      const char* unifrac_method, bool variance_adjust, double alpha,
                                      bool bypass_tips, unsigned int n_substeps, unsigned int stripe_start,
                                      unsigned int stripe_stop, partial_mat_t** result);

/* Compute UniFrac - against in-memory objects returning full form matrix
 *
 * biom_data <su_c_biom_inmem_t *> BIOM C data struct
 * tree_data <su_c_bptree_t *> BPTree C data struct
 * unifrac_method <const char*> the requested unifrac method.
 * variance_adjust <bool> whether to apply variance adjustment.
 * alpha <double> GUniFrac alpha, only relevant if method == generalized.
 * bypass_tips <bool> disregard tips, reduces compute by about 50%
 * n_substeps <uint> the number of substeps to use.
 * subsample_depth <uint> Depth of subsampling, if >0
 * subsample_with_replacement <bool> Use subsampling with replacement? (only True supported)
 * mmap_dir <const char*> If not NULL, area to use for temp memory storage
 * result <mat_full_fp64_t**> the resulting distance matrix in full form, this is initialized within the method so using **
 *
 * one_off_inmem returns the following error codes:
 *
 * okay           : no problems encountered
 * unknown_method : the requested method is unknown.
 * table_empty    : the table does not have any entries
 */
EXTERN ComputeStatus one_off_matrix_inmem_nv_fp64_v2(su_c_biom_inmem_t *biom_data, su_c_bptree_t *tree_data,
                                                     const char* unifrac_method, bool variance_adjust, double alpha,
                                                     bool bypass_tips, unsigned int n_substeps,
                                                     unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                     mat_full_fp64_t** result);

EXTERN ComputeStatus one_off_matrix_sparse_nv_fp64_v2(su_c_biom_sparse_t *biom_data, su_c_bptree_t *tree_data,
                                                      const char* unifrac_method, bool variance_adjust, double alpha,
                                                      bool bypass_tips, unsigned int n_substeps,
                                                      unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                      mat_full_fp64_t** result);

/* Compute UniFrac - against in-memory objects returning full form matrix, fp32
 *
 * biom_data <su_c_biom_inmem_t *> BIOM C data struct
 * tree_data <su_c_bptree_t *> BPTree C data struct
 * unifrac_method <const char*> the requested unifrac method.
 * variance_adjust <bool> whether to apply variance adjustment.
 * alpha <double> GUniFrac alpha, only relevant if method == generalized.
 * bypass_tips <bool> disregard tips, reduces compute by about 50%
 * n_substeps <uint> the number of substeps to use.
 * subsample_depth <uint> Depth of subsampling, if >0
 * subsample_with_replacement <bool> Use subsampling with replacement? (only True supported)
 * mmap_dir <const char*> If not NULL, area to use for temp memory storage
 * result <mat_full_fp32_t**> the resulting distance matrix in full form, this is initialized within the method so using **
 *
 * one_off_inmem returns the following error codes:
 *
 * okay           : no problems encountered
 * unknown_method : the requested method is unknown.
 * table_empty    : the table does not have any entries
 */
EXTERN ComputeStatus one_off_matrix_inmem_nv_fp32_v2(su_c_biom_inmem_t *biom_data, su_c_bptree_t *tree_data,
                                                     const char* unifrac_method, bool variance_adjust, double alpha,
                                                     bool bypass_tips, unsigned int n_substeps,
                                                     unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                     mat_full_fp32_t** result);

EXTERN ComputeStatus one_off_matrix_sparse_nv_fp32_v2(su_c_biom_sparse_t *biom_data, su_c_bptree_t *tree_data,
                                                      const char* unifrac_method, bool variance_adjust, double alpha,
                                                      bool bypass_tips, unsigned int n_substeps,
                                                      unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                      mat_full_fp32_t** result);

#endif
