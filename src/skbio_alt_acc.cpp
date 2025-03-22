/*
 * Classes, methods and unction that provide skbio-like unctionality
 */

#include "skbio_alt.hpp"
#include <stdio.h> 
#include <stdlib.h> 


//
// ======================= permanova ========================
//
#pragma omp requires unified_address
#pragma omp requires unified_shared_memory


// Compute PERMANOVA pseudo-F partial statistic
// mat is symmetric matrix of size n_dims x in_n
// grouping is an array of size in_n
// inv_group_sizes is an array of size maxel(grouping)
// TILE is the loop tiling parameter
template<class TRealIn, class TRealOut>
inline TRealOut permanova_f_stat_sW_T_one(const TRealIn * mat, const uint32_t n_dims,
                                  const uint32_t *grouping,
                                  const TRealOut *inv_group_sizes,
                                  const uint32_t TILE) {
  TRealOut s_W = 0.0;

#pragma omp parallel for collapse(2) reduction(+:s_W)
  for (uint32_t row=0; row < (n_dims-1); row++) {   // no columns in last row
    for (uint32_t col=row+1; col < n_dims; col++) { // diagonal is always zero
        uint32_t group_idx = grouping[row];
        const TRealIn * mat_row = mat + uint64_t(row)*uint64_t(n_dims);
        if (grouping[col] == group_idx) {
            TRealOut val = mat_row[col];
            s_W += val * val * inv_group_sizes[group_idx];;
        }
    }
  }

  return s_W;
}

// Compute PERMANOVA pseudo-F partial statistic
// mat is symmetric matrix of size n_dims x n_dims
// groupings is a matrix of size n_dims x n_grouping_dims
// inv_group_sizes is an array of size maxel(groupings)
// TILE is the loop tiling parameter
// Results in group_sWs, and array of size n_grouping_dims
// Note: Best results when TILE is about cache line and n_grouping_dims fits in L1 cache
template<class TRealIn, class TRealOut>
inline void permanova_f_stat_sW_T(const TRealIn * mat, const uint32_t n_dims,
                                  const uint32_t *groupings, const uint32_t n_grouping_dims,
                                  const TRealOut *inv_group_sizes,
                                  const uint32_t TILE,
                                  TRealOut *group_sWs) {
 //fprintf(stderr,"permanova_f_stat_sW IN (%i)\n", int(n_grouping_dims));
//pragma omp target teams distribute parallel for simd
#pragma omp target teams distribute
 for (uint32_t grouping_el=0; grouping_el < n_grouping_dims; grouping_el++) {
    const uint32_t *grouping = groupings + uint64_t(grouping_el)*uint64_t(n_dims);
    group_sWs[grouping_el] = permanova_f_stat_sW_T_one(mat,n_dims,grouping,inv_group_sizes,TILE);
 } 
 fprintf(stderr,"permanova_f_stat_sW OUT\n");
}

void permanova_f_stat_sW_fp32(const float * mat, const uint32_t n_dims,
                                  const uint32_t *groupings, const uint32_t n_grouping_dims,
                                  const float *inv_group_sizes,
                                  const uint32_t TILE,
                                  float *group_sWs) {
  permanova_f_stat_sW_T(mat,n_dims,groupings,n_grouping_dims,inv_group_sizes,TILE,group_sWs);
}

void permanova_f_stat_sW_fp64(const double * mat, const uint32_t n_dims,
                                  const uint32_t *groupings, const uint32_t n_grouping_dims,
                                  const double *inv_group_sizes,
                                  const uint32_t TILE,
                                  double *group_sWs) {
  permanova_f_stat_sW_T(mat,n_dims,groupings,n_grouping_dims,inv_group_sizes,TILE,group_sWs);
}


void permanova_f_stat_sW_fpmixed(const float * mat, const uint32_t n_dims,
                                  const uint32_t *groupings, const uint32_t n_grouping_dims,
                                  const double *inv_group_sizes,
                                  const uint32_t TILE,
                                  double *group_sWs) {
  permanova_f_stat_sW_T(mat,n_dims,groupings,n_grouping_dims,inv_group_sizes,TILE,group_sWs);
}


