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
 * biom_filename <const char*> the filename to the biom table.
 * tree_filename <const char*> the filename to the correspodning tree.
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
 * table_missing  : the filename for the table does not exist
 * tree_missing   : the filename for the tree does not exist
 * unknown_method : the requested method is unknown.
 * table_empty    : the table does not have any entries
 */
EXTERN ComputeStatus one_off_nv_fp64(const char* biom_filename, const char* tree_filename,
                                     const char* unifrac_method, bool variance_adjust, double alpha,
                                     bool bypass_tips, unsigned int n_substeps, mat_t** result);


/* Compute UniFrac - against in-memory objects returning full form matrix
 *
 * table <biom> a constructed BIOM object
 * tree <BPTree> a constructed BPTree object
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
EXTERN ComputeStatus one_off_matrix_inmem_nv_fp64_v2(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                                     const char* unifrac_method, bool variance_adjust, double alpha,
                                                     bool bypass_tips, unsigned int n_substeps,
                                                     unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                     mat_full_fp64_t** result);

/* Compute UniFrac - against in-memory objects returning full form matrix, fp32
 *
 * table <biom> a constructed BIOM object
 * tree <BPTree> a constructed BPTree object
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
EXTERN ComputeStatus one_off_matrix_inmem_nv_fp32_v2(const support_biom_t *table_data, const support_bptree_t *tree_data,
                                                     const char* unifrac_method, bool variance_adjust, double alpha,
                                                     bool bypass_tips, unsigned int n_substeps,
                                                     unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                                     mat_full_fp32_t** result);

/* Compute UniFrac - matrix form
 *
 * biom_filename <const char*> the filename to the biom table.
 * tree_filename <const char*> the filename to the correspodning tree.
 * unifrac_method <const char*> the requested unifrac method.
 * variance_adjust <bool> whether to apply variance adjustment.
 * alpha <double> GUniFrac alpha, only relevant if method == generalized.
 * bypass_tips <bool> disregard tips, reduces compute by about 50%
 * n_substeps <uint> the number of substeps/blocks to use.
 * subsample_depth <uint> Depth of subsampling, if >0
 * subsample_with_replacement <bool> Use subsampling with replacement? (only True supported)
 * mmap_dir <const char*> If not NULL, area to use for temp memory storage
 * result <mat_full_fp64_t**> the resulting distance matrix in matrix form, this is initialized within the method so using **
 *
 * one_off_matrix returns the following error codes:
 *
 * okay           : no problems encountered
 * table_missing  : the filename for the table does not exist
 * tree_missing   : the filename for the tree does not exist
 * unknown_method : the requested method is unknown.
 * table_empty    : the table does not have any entries
 */
EXTERN ComputeStatus one_off_matrix_nv_fp64_v2(const char* biom_filename, const char* tree_filename,
                                       const char* unifrac_method, bool variance_adjust, double alpha,
                                       bool bypass_tips, unsigned int n_substeps,
                                       unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                       mat_full_fp64_t** result);


/* Compute UniFrac - matrix form, fp32 variant
 *
 * biom_filename <const char*> the filename to the biom table.
 * tree_filename <const char*> the filename to the correspodning tree.
 * unifrac_method <const char*> the requested unifrac method.
 * variance_adjust <bool> whether to apply variance adjustment.
 * alpha <double> GUniFrac alpha, only relevant if method == generalized.
 * bypass_tips <bool> disregard tips, reduces compute by about 50%
 * n_substeps <uint> the number of substeps/blocks to use.
 * subsample_depth <uint> Depth of subsampling, if >0
 * subsample_with_replacement <bool> Use subsampling with replacement? (only True supported)
 * mmap_dir <const char*> If not NULL, area to use for temp memory storage
 * result <mat_full_fp32_t**> the resulting distance matrix in matrix form, this is initialized within the method so using **
 *
 * one_off_matrix_fp32 returns the following error codes:
 *
 * okay           : no problems encountered
 * table_missing  : the filename for the table does not exist
 * tree_missing   : the filename for the tree does not exist
 * unknown_method : the requested method is unknown.
 * table_empty    : the table does not have any entries
 */
EXTERN ComputeStatus one_off_matrix_nv_fp32_v2(const char* biom_filename, const char* tree_filename,
                                            const char* unifrac_method, bool variance_adjust, double alpha,
                                            bool bypass_tips, unsigned int n_substeps,
                                            unsigned int subsample_depth, bool subsample_with_replacement, const char *mmap_dir,
                                            mat_full_fp32_t** result);

#endif
