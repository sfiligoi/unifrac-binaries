#include "unifrac_task_noclass.hpp"

#include "unifrac_task_impl.hpp"

/*
 * Create exportable variants
 */

bool SUCMP_NM::acc_found_gpu() {
  return acc_found_gpu_T();
}

// is the implementation async, and need the alt structures?
bool SUCMP_NM::acc_need_alt() {
  return acc_need_alt_T();
}

void SUCMP_NM::acc_wait() {
  acc_wait_T();
}

/*
 * Create concrete implementation wrappers
 */

template<>
void SUCMP_NM::acc_create_buf(float *buf, uint64_t start, uint64_t end) {
   acc_create_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_create_buf(double *buf, uint64_t start, uint64_t end) {
   acc_create_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_create_buf(uint64_t *buf, uint64_t start, uint64_t end) {
   acc_create_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_create_buf(uint32_t *buf, uint64_t start, uint64_t end) {
   acc_create_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_create_buf(bool *buf, uint64_t start, uint64_t end) {
   acc_create_buf_T(buf, start, end);
}

// ==================================

template<>
void SUCMP_NM::acc_copyin_buf(float *buf, uint64_t start, uint64_t end) {
   acc_copyin_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_copyin_buf(double *buf, uint64_t start, uint64_t end) {
   acc_copyin_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_copyin_buf(uint64_t *buf, uint64_t start, uint64_t end) {
   acc_copyin_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_copyin_buf(uint32_t *buf, uint64_t start, uint64_t end) {
   acc_copyin_buf_T(buf, start, end);
}

// ==================================

template<>
void SUCMP_NM::acc_update_device(float *buf, uint64_t start, uint64_t end) {
   acc_update_device_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_update_device(double *buf, uint64_t start, uint64_t end) {
   acc_update_device_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_update_device(uint32_t *buf, uint64_t start, uint64_t end) {
   acc_update_device_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_update_device(uint64_t *buf, uint64_t start, uint64_t end) {
   acc_update_device_T(buf, start, end);
}

// ==================================

template<>
void SUCMP_NM::acc_copyout_buf(float *buf, uint64_t start, uint64_t end) {
   acc_copyout_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_copyout_buf(double *buf, uint64_t start, uint64_t end) {
   acc_copyout_buf_T(buf, start, end);
}

// ==================================

template<>
void SUCMP_NM::acc_destroy_buf(float *buf, uint64_t start, uint64_t end) {
   acc_destroy_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_destroy_buf(double *buf, uint64_t start, uint64_t end) {
   acc_destroy_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_destroy_buf(uint32_t *buf, uint64_t start, uint64_t end) {
   acc_destroy_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_destroy_buf(uint64_t *buf, uint64_t start, uint64_t end) {
   acc_destroy_buf_T(buf, start, end);
}
template<>
void SUCMP_NM::acc_destroy_buf(bool *buf, uint64_t start, uint64_t end) {
   acc_destroy_buf_T(buf, start, end);
}

// ==================================

template<>
void SUCMP_NM::compute_stripes_totals(
		float * const __restrict__ dm_stripes_buf,
		const float * const __restrict__ dm_stripes_total_buf,
		const uint64_t bufels) {
   compute_stripes_totals_T(dm_stripes_buf, dm_stripes_total_buf, bufels);
}

template<>
void SUCMP_NM::compute_stripes_totals(
		double * const __restrict__ dm_stripes_buf,
		const double * const __restrict__ dm_stripes_total_buf,
		const uint64_t bufels) {
   compute_stripes_totals_T(dm_stripes_buf, dm_stripes_total_buf, bufels);
}

// ==================================

template<>
void SUCMP_NM::run_UnnormalizedWeightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const float * const __restrict__ embedded_proportions,
		float * const __restrict__ dm_stripes_buf,
		bool * const __restrict__ zcheck,
		float * const __restrict__ sums) {
   run_UnnormalizedWeightedTask_T(
		   filled_embs,
		   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf,
		   zcheck, sums);
}

template<>
void SUCMP_NM::run_UnnormalizedWeightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const double * const __restrict__ embedded_proportions,
		double * const __restrict__ dm_stripes_buf,
		bool * const __restrict__ zcheck,
		double * const __restrict__ sums) {
   run_UnnormalizedWeightedTask_T(
		   filled_embs,
		   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf,
		   zcheck, sums);
}

// ==================================

template<>
void SUCMP_NM::run_VawUnnormalizedWeightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const float * const __restrict__ embedded_proportions,
		const float * const __restrict__ embedded_counts,
		const float * const __restrict__ sample_total_counts,
		float * const __restrict__ dm_stripes_buf) {
   run_VawUnnormalizedWeightedTask_T(
		   filled_embs,
		   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf);
}

template<>
void SUCMP_NM::run_VawUnnormalizedWeightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const double * const __restrict__ embedded_proportions,
		const double * const __restrict__ embedded_counts,
		const double * const __restrict__ sample_total_counts,
		double * const __restrict__ dm_stripes_buf) {
   run_VawUnnormalizedWeightedTask_T(
		   filled_embs,
		   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf);
}

// ==================================

template<>
void SUCMP_NM::run_NormalizedWeightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const float * const __restrict__ embedded_proportions,
		float * const __restrict__ dm_stripes_buf,
		float * const __restrict__ dm_stripes_total_buf,
		bool * const __restrict__ zcheck,
		float * const __restrict__ sums) {
   run_NormalizedWeightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf, dm_stripes_total_buf,
		   zcheck, sums);
}

template<>
void SUCMP_NM::run_NormalizedWeightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const double * const __restrict__ embedded_proportions,
		double * const __restrict__ dm_stripes_buf,
		double * const __restrict__ dm_stripes_total_buf,
		bool * const __restrict__ zcheck,
		double * const __restrict__ sums) {
   run_NormalizedWeightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf, dm_stripes_total_buf,
		   zcheck, sums);
}

// ==================================

template<>
void SUCMP_NM::run_VawNormalizedWeightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const float * const __restrict__ embedded_proportions,
		const float * const __restrict__ embedded_counts,
		const float * const __restrict__ sample_total_counts,
		float * const __restrict__ dm_stripes_buf,
		float * const __restrict__ dm_stripes_total_buf) {
   run_VawNormalizedWeightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf, dm_stripes_total_buf);
}

template<>
void SUCMP_NM::run_VawNormalizedWeightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const double * const __restrict__ embedded_proportions,
		const double * const __restrict__ embedded_counts,
		const double * const __restrict__ sample_total_counts,
		double * const __restrict__ dm_stripes_buf,
		double * const __restrict__ dm_stripes_total_buf) {
   run_VawNormalizedWeightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf, dm_stripes_total_buf);
}

// ==================================

template<>
void SUCMP_NM::run_GeneralizedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const float * const __restrict__ embedded_proportions,
		float * const __restrict__ dm_stripes_buf,
		float * const __restrict__ dm_stripes_total_buf,
		const float g_unifrac_alpha) {
   run_GeneralizedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf, dm_stripes_total_buf,
		   g_unifrac_alpha);
}

template<>
void SUCMP_NM::run_GeneralizedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const double * const __restrict__ embedded_proportions,
		double * const __restrict__ dm_stripes_buf,
		double * const __restrict__ dm_stripes_total_buf,
		const double g_unifrac_alpha) {
   run_GeneralizedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf, dm_stripes_total_buf,
		   g_unifrac_alpha);
}

// ==================================

template<>
void SUCMP_NM::run_VawGeneralizedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const float * const __restrict__ embedded_proportions,
		const float * const __restrict__ embedded_counts,
		const float * const __restrict__ sample_total_counts ,
		float * const __restrict__ dm_stripes_buf,
		float * const __restrict__ dm_stripes_total_buf,
		const float g_unifrac_alpha) {
   run_VawGeneralizedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf, dm_stripes_total_buf,
		   g_unifrac_alpha);
}

template<>
void SUCMP_NM::run_VawGeneralizedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const double * const __restrict__ embedded_proportions,
		const double * const __restrict__ embedded_counts,
		const double * const __restrict__ sample_total_counts ,
		double * const __restrict__ dm_stripes_buf,
		double * const __restrict__ dm_stripes_total_buf,
		const double g_unifrac_alpha) {
   run_VawGeneralizedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf, dm_stripes_total_buf,
		   g_unifrac_alpha);
}

// ==================================

template<>
void SUCMP_NM::run_UnweightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const uint64_t * const __restrict__ embedded_proportions,
		float * const __restrict__ dm_stripes_buf,
		float * const __restrict__ dm_stripes_total_buf,
		float * const __restrict__ sums,
		bool   * const __restrict__ zcheck,
		float * const __restrict__ stripe_sums) {
   run_UnweightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf, dm_stripes_total_buf,
		   sums, zcheck, stripe_sums);
}

template<>
void SUCMP_NM::run_UnweightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const uint64_t * const __restrict__ embedded_proportions,
		double * const __restrict__ dm_stripes_buf,
		double * const __restrict__ dm_stripes_total_buf,
		double * const __restrict__ sums,
		bool   * const __restrict__ zcheck,
		double * const __restrict__ stripe_sums) {
   run_UnweightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf, dm_stripes_total_buf,
		   sums, zcheck, stripe_sums);
}

// ==================================

template<>
void SUCMP_NM::run_UnnormalizedUnweightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const uint64_t * const __restrict__ embedded_proportions,
		float * const __restrict__ dm_stripes_buf,
		float * const __restrict__ sums,
		bool   * const __restrict__ zcheck,
		float * const __restrict__ stripe_sums) {
   run_UnnormalizedUnweightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf,
		   sums, zcheck, stripe_sums);
}

template<>
void SUCMP_NM::run_UnnormalizedUnweightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const uint64_t * const __restrict__ embedded_proportions,
		double * const __restrict__ dm_stripes_buf,
		double * const __restrict__ sums,
		bool   * const __restrict__ zcheck,
		double * const __restrict__ stripe_sums) {
   run_UnnormalizedUnweightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions,
		   dm_stripes_buf,
		   sums, zcheck, stripe_sums);
}

// ==================================

template<>
void SUCMP_NM::run_VawUnweightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const uint32_t * const __restrict__ embedded_proportions,
		const float  * const __restrict__ embedded_counts,
		const float  * const __restrict__ sample_total_counts,
		float * const __restrict__ dm_stripes_buf,
		float * const __restrict__ dm_stripes_total_buf) {
   run_VawUnweightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf, dm_stripes_total_buf);
}

template<>
void SUCMP_NM::run_VawUnweightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const uint32_t * const __restrict__ embedded_proportions,
		const double  * const __restrict__ embedded_counts,
		const double  * const __restrict__ sample_total_counts,
		double * const __restrict__ dm_stripes_buf,
		double * const __restrict__ dm_stripes_total_buf) {
   run_VawUnweightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf, dm_stripes_total_buf);
}

// ==================================

template<>
void SUCMP_NM::run_VawUnnormalizedUnweightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const float * const __restrict__ lengths,
		const uint32_t * const __restrict__ embedded_proportions,
		const float  * const __restrict__ embedded_counts,
		const float  * const __restrict__ sample_total_counts,
		float * const __restrict__ dm_stripes_buf) {
   run_VawUnnormalizedUnweightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf);
}

template<>
void SUCMP_NM::run_VawUnnormalizedUnweightedTask(
		const unsigned int filled_embs,
		const uint64_t start_idx, const uint64_t stop_idx,
		const uint64_t n_samples, const uint64_t n_samples_r,
		const double * const __restrict__ lengths,
		const uint32_t * const __restrict__ embedded_proportions,
		const double  * const __restrict__ embedded_counts,
		const double  * const __restrict__ sample_total_counts,
		double * const __restrict__ dm_stripes_buf) {
   run_VawUnnormalizedUnweightedTask_T(
		   filled_embs,
                   start_idx, stop_idx, n_samples, n_samples_r,
		   lengths, embedded_proportions, embedded_counts, sample_total_counts,
		   dm_stripes_buf);
}

