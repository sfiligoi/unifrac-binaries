/*
 * Classes, methods and unction that provide skbio-like unctionality
 */

#include "skbio_alt.hpp"

//
// ======================= permanova ========================
//

static void (*dl_permanova_f_stat_sW_fp32)(const float * , const uint32_t ,
                                  const uint32_t *, const uint32_t ,
                                  const float *,
                                  const uint32_t ,
                                  float *) = NULL;
static void (*dl_permanova_f_stat_sW_fp64)(const double * , const uint32_t ,
                                  const uint32_t *, const uint32_t ,
                                  const double *,
                                  const uint32_t ,
                                  double *) = NULL;
static void (*dl_permanova_f_stat_sW_fpmixed)(const float * , const uint32_t ,
                                  const uint32_t *, const uint32_t ,
                                  const double *,
                                  const uint32_t ,
                                  double *) = NULL;


void permanova_f_stat_sW_fp32(const float * mat, const uint32_t n_dims,
                                  const uint32_t *groupings, const uint32_t n_grouping_dims,
                                  const float *inv_group_sizes,
                                  const uint32_t TILE,
                                  float *group_sWs) {
   cond_ssu_load("permanova_f_stat_sW_fp32", (void **) &dl_permanova_f_stat_sW_fp32);

   (*dl_permanova_f_stat_sW_fp32)(mat, n_dims, groupings, n_grouping_dims, inv_group_sizes, TILE, group_sWs);
}

void permanova_f_stat_sW_fp64(const double * mat, const uint32_t n_dims,
                                  const uint32_t *groupings, const uint32_t n_grouping_dims,
                                  const double *inv_group_sizes,
                                  const uint32_t TILE,
                                  double *group_sWs) {
   cond_ssu_load("permanova_f_stat_sW_fp64", (void **) &dl_permanova_f_stat_sW_fp64);

   (*dl_permanova_f_stat_sW_fp64)(mat, n_dims, groupings, n_grouping_dims, inv_group_sizes, TILE, group_sWs);
}


void permanova_f_stat_sW_fpmixed(const float * mat, const uint32_t n_dims,
                                  const uint32_t *groupings, const uint32_t n_grouping_dims,
                                  const double *inv_group_sizes,
                                  const uint32_t TILE,
                                  double *group_sWs) {
   cond_ssu_load("permanova_f_stat_sW_fpmixed", (void **) &dl_permanova_f_stat_sW_fpmixed);

   (*dl_permanova_f_stat_sW_fpmixed)(mat, n_dims, groupings, n_grouping_dims, inv_group_sizes, TILE, group_sWs);
}


