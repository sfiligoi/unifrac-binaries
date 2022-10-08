#include <algorithm> 
#include "unifrac_task.hpp"
#include <cstdlib>

#ifndef _OPENACC
// popcnt returns number of bits set to 1, if supported by fast HW compute
// else return 0 when v is 0 and max-bits else
// Note: The user of popcnt in this file must be able to use this modified semantics

#if __AVX__
#include <immintrin.h>
static inline int32_t popcnt_u32(uint32_t v) {return  _mm_popcnt_u32(v);}
static inline int64_t popcnt_u64(uint64_t v) {return  _mm_popcnt_u64(v);}
#else
static inline int32_t popcnt_u32(uint32_t v) {return (v==0) ? 0 : 32;}
static inline int64_t popcnt_u64(uint64_t v) {return (v==0) ? 0 : 64;}
// For future non-x86 ports, see https://barakmich.dev/posts/popcnt-arm64-go-asm/
// and https://stackoverflow.com/questions/38113284/whats-the-difference-between-builtin-popcountll-and-mm-popcnt-u64
#endif
#endif

// check for zero values and pre-compute single column sums
template<class TFloat>
static inline void WeightedZerosAndSums(
                      bool   * const __restrict__ zcheck,
                      TFloat * const __restrict__ sums,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t n_samples,
                      const uint64_t n_samples_r) {
#ifdef _OPENACC
#pragma acc parallel loop gang vector present(embedded_proportions,lengths,zcheck,sums)
#else
#pragma omp parallel for default(shared)
#endif
    for(uint64_t k=0; k<n_samples; k++) {
            bool all_zeros=true;
            TFloat my_sum = 0.0;

#pragma acc loop seq
            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r * emb;

                TFloat u1 = embedded_proportions[offset + k];
                my_sum += u1*lengths[emb];
                all_zeros = all_zeros && (u1==0.0);
            }

            sums[k]     = my_sum;
            zcheck[k] = all_zeros;
    }
}


// check for zero values
template<class T>
static inline void WeightedZerosAsync(
                      bool   * const __restrict__ zcheck,
                      const T * const __restrict__ embedded_proportions,
                      const unsigned int filled_embs,
                      const uint64_t n_samples,
                      const uint64_t n_samples_r) {
#ifdef _OPENACC
#pragma acc parallel loop gang vector present(embedded_proportions,zcheck) async
#else
#pragma omp parallel for default(shared)
#endif
    for(uint64_t k=0; k<n_samples; k++) {
            bool all_zeros=true;

#pragma acc loop seq
            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r * emb;

                T u1 = embedded_proportions[offset + k];
                all_zeros = all_zeros && (u1==0);
            }

            zcheck[k] = all_zeros;
    }
}


// Single step in computing Weighted part of Unifrac
template<class TFloat>
static inline TFloat WeightedVal1(
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t n_samples_r,
                      const uint64_t k,
                      const uint64_t l1) {
            TFloat my_stripe = 0.0;

            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r * emb;

                TFloat u1 = embedded_proportions[offset + k];
                TFloat v1 = embedded_proportions[offset + l1];
                TFloat diff1 = u1 - v1;
                TFloat length = lengths[emb];

                my_stripe     += fabs(diff1) * length;
            } // for emb

            return my_stripe;
}

#ifndef _OPENACC
template<class TFloat>
static inline void WeightedVal4(
                      TFloat * const __restrict__ stripes,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t n_samples_r,
                      const uint64_t ks,
                      const uint64_t ls) {
            TFloat my_stripe0 = 0.0;
            TFloat my_stripe1 = 0.0;
            TFloat my_stripe2 = 0.0;
            TFloat my_stripe3 = 0.0;

            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r * emb;

                TFloat u0 = embedded_proportions[offset + ks];
                TFloat u1 = embedded_proportions[offset + ks + 1];
                TFloat u2 = embedded_proportions[offset + ks + 2];
                TFloat u3 = embedded_proportions[offset + ks + 3];
                TFloat v0 = embedded_proportions[offset + ls];
                TFloat v1 = embedded_proportions[offset + ls + 1];
                TFloat v2 = embedded_proportions[offset + ls + 2];
                TFloat v3 = embedded_proportions[offset + ls + 3];
                TFloat length = lengths[emb];

                TFloat diff0 = u0 - v0;
                TFloat diff1 = u1 - v1;
                TFloat diff2 = u2 - v2;
                TFloat diff3 = u3 - v3;
                TFloat absdiff0 = fabs(diff0);
                TFloat absdiff1 = fabs(diff1);
                TFloat absdiff2 = fabs(diff2);
                TFloat absdiff3 = fabs(diff3);

                my_stripe0     += absdiff0 * length;
                my_stripe1     += absdiff1 * length;
                my_stripe2     += absdiff2 * length;
                my_stripe3     += absdiff3 * length;
            } // for emb

            stripes[ks] += my_stripe0;
            stripes[ks + 1] += my_stripe1;
            stripes[ks + 2] += my_stripe2;
            stripes[ks + 3] += my_stripe3;
}

template<class TFloat>
static inline void WeightedVal8(
                      TFloat * const __restrict__ stripes,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t n_samples_r,
                      const uint64_t ks,
                      const uint64_t ls) {
            TFloat my_stripe0 = 0.0;
            TFloat my_stripe1 = 0.0;
            TFloat my_stripe2 = 0.0;
            TFloat my_stripe3 = 0.0;
            TFloat my_stripe4 = 0.0;
            TFloat my_stripe5 = 0.0;
            TFloat my_stripe6 = 0.0;
            TFloat my_stripe7 = 0.0;

            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r * emb;

                TFloat u0 = embedded_proportions[offset + ks];
                TFloat u1 = embedded_proportions[offset + ks + 1];
                TFloat u2 = embedded_proportions[offset + ks + 2];
                TFloat u3 = embedded_proportions[offset + ks + 3];
                TFloat u4 = embedded_proportions[offset + ks + 4];
                TFloat u5 = embedded_proportions[offset + ks + 5];
                TFloat u6 = embedded_proportions[offset + ks + 6];
                TFloat u7 = embedded_proportions[offset + ks + 7];
                TFloat v0 = embedded_proportions[offset + ls];
                TFloat v1 = embedded_proportions[offset + ls + 1];
                TFloat v2 = embedded_proportions[offset + ls + 2];
                TFloat v3 = embedded_proportions[offset + ls + 3];
                TFloat v4 = embedded_proportions[offset + ls + 4];
                TFloat v5 = embedded_proportions[offset + ls + 5];
                TFloat v6 = embedded_proportions[offset + ls + 6];
                TFloat v7 = embedded_proportions[offset + ls + 7];
                TFloat length = lengths[emb];

                TFloat diff0 = u0 - v0;
                TFloat diff1 = u1 - v1;
                TFloat diff2 = u2 - v2;
                TFloat diff3 = u3 - v3;
                TFloat diff4 = u4 - v4;
                TFloat diff5 = u5 - v5;
                TFloat diff6 = u6 - v6;
                TFloat diff7 = u7 - v7;
                TFloat absdiff0 = fabs(diff0);
                TFloat absdiff1 = fabs(diff1);
                TFloat absdiff2 = fabs(diff2);
                TFloat absdiff3 = fabs(diff3);
                TFloat absdiff4 = fabs(diff4);
                TFloat absdiff5 = fabs(diff5);
                TFloat absdiff6 = fabs(diff6);
                TFloat absdiff7 = fabs(diff7);

                my_stripe0     += absdiff0 * length;
                my_stripe1     += absdiff1 * length;
                my_stripe2     += absdiff2 * length;
                my_stripe3     += absdiff3 * length;
                my_stripe4     += absdiff4 * length;
                my_stripe5     += absdiff5 * length;
                my_stripe6     += absdiff6 * length;
                my_stripe7     += absdiff7 * length;
            } // for emb

            stripes[ks] += my_stripe0;
            stripes[ks + 1] += my_stripe1;
            stripes[ks + 2] += my_stripe2;
            stripes[ks + 3] += my_stripe3;
            stripes[ks + 4] += my_stripe4;
            stripes[ks + 5] += my_stripe5;
            stripes[ks + 6] += my_stripe6;
            stripes[ks + 7] += my_stripe7;
}
#endif

// Single step in computing NormalizedWeighted Unifrac
template<class TFloat>
static inline void UnnormalizedWeighted1(
                      TFloat * const __restrict__ dm_stripes_buf,
                      const bool   * const __restrict__ zcheck,
                      const TFloat * const __restrict__ sums,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t idx,
                      const uint64_t n_samples_r,
                      const uint64_t k,
                      const uint64_t l1) {
       const bool allzero_k = zcheck[k];
       const bool allzero_l1 = zcheck[l1];

       if (allzero_k && allzero_l1) {
         // nothing to do, would have to add 0
       } else {
          TFloat my_stripe;

          if (allzero_k || allzero_l1) {
            // one side has all zeros
            // we can use the distributed property, and use the pre-computed values

            const uint64_t ridx = (allzero_k) ? l1 : // if (nonzero_l1) ridx=l1 // fabs(k-l1), with k==0
                                                k;   // if (nonzero_k)  ridx=k  // fabs(k-l1), with l1==0

            // keep reads in the same place to maximize GPU warp performance
            my_stripe = sums[ridx];
          } else {
            // both sides non zero, use the explicit but slow approach
            my_stripe = WeightedVal1(embedded_proportions, lengths,
                                     filled_embs, n_samples_r,
                                     k, l1);
          }

          TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
          //TFloat *dm_stripe = dm_stripes[stripe];
          
          // keep all writes in a single place, to maximize GPU warp performance
          dm_stripe[k]       += my_stripe;
       }
}

#ifndef _OPENACC
// Vectorized step in computing UnnormalizedWeighted Unifrac
template<class TFloat>
static inline void UnnormalizedWeighted4(
                      TFloat * const __restrict__ dm_stripes_buf,
                      const bool   * const __restrict__ zcheck,
                      const TFloat * const __restrict__ sums,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t idx,
                      const uint64_t n_samples_r,
                      const uint64_t ks,
                      const uint64_t ls) {
       const uint32_t z_k = ((const uint32_t *)(zcheck+ks))[0];
       const uint32_t z_l = ((const uint32_t *)(zcheck+ls))[0];
       const bool allzero_k = z_k==0x01010101;
       const bool allzero_l = z_l==0x01010101;

       // popcnt is cheap but has large latency, so compute speculatively/early
       const int32_t cnt_k = popcnt_u32(z_k);
       const int32_t cnt_l = popcnt_u32(z_l);

       if (allzero_k && allzero_l) {
         // nothing to do, would have to add 0
       } else if (allzero_k || allzero_l) {
          TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
          //TFloat *dm_stripe = dm_stripes[stripe];

          if (allzero_k) {
             // one side has all zeros, use distributed property
             // if (nonzero_l1) ridx=l1 // fabs(k-l1), with k==0
             const TFloat sum_l0 = sums[ls];
             const TFloat sum_l1 = sums[ls+1];
             const TFloat sum_l2 = sums[ls+2];
             const TFloat sum_l3 = sums[ls+3];
             dm_stripe[ks]   += sum_l0;
             dm_stripe[ks+1] += sum_l1;
             dm_stripe[ks+2] += sum_l2;
             dm_stripe[ks+3] += sum_l3;
          } else {
             // one side has all zeros, use distributed property
             // if (nonzero_k)  ridx=k  // fabs(k-l1), with l1==0
             const TFloat sum_k0 = sums[ks];
             const TFloat sum_k1 = sums[ks+1];
             const TFloat sum_k2 = sums[ks+2];
             const TFloat sum_k3 = sums[ks+3];
             dm_stripe[ks]   += sum_k0;
             dm_stripe[ks+1] += sum_k1;
             dm_stripe[ks+2] += sum_k2;
             dm_stripe[ks+3] += sum_k3;
          }
       } else if ((cnt_k<3) && (cnt_l<3)) {
          // several of the elemens are nonzero, may as well use the vectorized version
          TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
          //TFloat *dm_stripe = dm_stripes[stripe];

          WeightedVal4(dm_stripe,
                       embedded_proportions, lengths,
                       filled_embs, n_samples_r,
                       ks, ls);
       } else {
         // only a few have both sides partially non zero, use the fine grained compute
         for (uint64_t i=0; i<4; i++) {
            UnnormalizedWeighted1<TFloat>(
                                   dm_stripes_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks+i, ls+i);
         }
       } // (allzero_k && allzero_l)
}

template<class TFloat>
static inline void UnnormalizedWeighted8(
                      TFloat * const __restrict__ dm_stripes_buf,
                      const bool   * const __restrict__ zcheck,
                      const TFloat * const __restrict__ sums,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t idx,
                      const uint64_t n_samples_r,
                      const uint64_t ks,
                      const uint64_t ls) {
       const uint64_t z_k = ((const uint64_t *)(zcheck+ks))[0];
       const uint64_t z_l = ((const uint64_t *)(zcheck+ls))[0];
       const bool allzero_k = z_k==0x0101010101010101;
       const bool allzero_l = z_l==0x0101010101010101;

       // popcnt is cheap but has large latency, so compute speculatively/early
       const int64_t cnt_k = popcnt_u64(z_k);
       const int64_t cnt_l = popcnt_u64(z_l);

       if (allzero_k && allzero_l) {
         // nothing to do, would have to add 0
       } else if (allzero_k || allzero_l) {
          TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
          //TFloat *dm_stripe = dm_stripes[stripe];

          if (allzero_k) {
             // one side has all zeros, use distributed property
             // if (nonzero_l1) ridx=l1 // fabs(k-l1), with k==0
             const TFloat sum_l0 = sums[ls];
             const TFloat sum_l1 = sums[ls+1];
             const TFloat sum_l2 = sums[ls+2];
             const TFloat sum_l3 = sums[ls+3];
             const TFloat sum_l4 = sums[ls+4];
             const TFloat sum_l5 = sums[ls+5];
             const TFloat sum_l6 = sums[ls+6];
             const TFloat sum_l7 = sums[ls+7];
             dm_stripe[ks]   += sum_l0;
             dm_stripe[ks+1] += sum_l1;
             dm_stripe[ks+2] += sum_l2;
             dm_stripe[ks+3] += sum_l3;
             dm_stripe[ks+4] += sum_l4;
             dm_stripe[ks+5] += sum_l5;
             dm_stripe[ks+6] += sum_l6;
             dm_stripe[ks+7] += sum_l7;
          } else {
             // one side has all zeros, use distributed property
             // if (nonzero_k)  ridx=k  // fabs(k-l1), with l1==0
             const TFloat sum_k0 = sums[ks];
             const TFloat sum_k1 = sums[ks+1];
             const TFloat sum_k2 = sums[ks+2];
             const TFloat sum_k3 = sums[ks+3];
             const TFloat sum_k4 = sums[ks+4];
             const TFloat sum_k5 = sums[ks+5];
             const TFloat sum_k6 = sums[ks+6];
             const TFloat sum_k7 = sums[ks+7];
             dm_stripe[ks]   += sum_k0;
             dm_stripe[ks+1] += sum_k1;
             dm_stripe[ks+2] += sum_k2;
             dm_stripe[ks+3] += sum_k3;
             dm_stripe[ks+4] += sum_k4;
             dm_stripe[ks+5] += sum_k5;
             dm_stripe[ks+6] += sum_k6;
             dm_stripe[ks+7] += sum_k7;
          } 
       } else if ((cnt_k<6) && (cnt_l<6)) {
          // several of the elemens are nonzero, may as well use the vectorized version
          TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
          //TFloat *dm_stripe = dm_stripes[stripe];

          WeightedVal8(dm_stripe,
                       embedded_proportions, lengths,
                       filled_embs, n_samples_r,
                       ks, ls);
       } else {
         // only a few have both sides partially non zero, use the fine grained compute
         for (uint64_t i=0; i<8; i++) {
            UnnormalizedWeighted1<TFloat>(
                                   dm_stripes_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks+i, ls+i);
         }
       } // (allzero_k && allzero_l)
}
#endif

template<class TFloat>
void SUCMP_NM::UnifracUnnormalizedWeightedTask<TFloat>::_run(unsigned int filled_embs, const TFloat * __restrict__ lengths) {
    const uint64_t start_idx = this->task_p->start;
    const uint64_t stop_idx = this->task_p->stop;
    const uint64_t n_samples = this->task_p->n_samples;
    const uint64_t n_samples_r = this->dm_stripes.n_samples_r;

    // openacc only works well with local variables
    const TFloat * const __restrict__ embedded_proportions = this->embedded_proportions;
    TFloat * const __restrict__ dm_stripes_buf = this->dm_stripes.buf;

    bool * const __restrict__ zcheck = this->zcheck;
    TFloat * const __restrict__ sums = this->sums;

    const uint64_t step_size = SUCMP_NM::UnifracUnnormalizedWeightedTask<TFloat>::step_size;
    const uint64_t sample_steps = (n_samples+(step_size-1))/step_size; // round up

    // check for zero values and pre-compute single column sums
    WeightedZerosAndSums(zcheck, sums,
                         embedded_proportions, lengths,
                         filled_embs, n_samples, n_samples_r);

    // now do the real compute
#ifdef _OPENACC
    const unsigned int acc_vector_size = SUCMP_NM::UnifracUnnormalizedWeightedTask<TFloat>::acc_vector_size;
#pragma acc parallel loop gang vector collapse(3) vector_length(acc_vector_size) present(embedded_proportions,dm_stripes_buf,lengths,zcheck,sums) async
#else
    // use dynamic scheduling due to non-homogeneity in the loop
#pragma omp parallel for default(shared) schedule(dynamic,1)
#endif
    for(uint64_t sk = 0; sk < sample_steps ; sk++) {
     for(uint64_t stripe = start_idx; stripe < stop_idx; stripe++) {
#ifdef _OPENACC
      // SIMT-based GPU work great one at a time (HW will deal with parallelism)
      for(uint64_t ik = 0; ik < step_size ; ik++) {
       const uint64_t k = sk*step_size + ik;

       if (k>=n_samples) continue; // past the limit

       const uint64_t l1 = (k + stripe + 1)%n_samples; // wraparound
       const uint64_t idx = (stripe-start_idx) * n_samples_r;

       UnnormalizedWeighted1<TFloat>(
                                   dm_stripes_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   k, l1);
      } // for ik
#else
      // SIMD-based CPUs need help with vectorization
      const uint64_t idx = (stripe-start_idx) * n_samples_r;
      uint64_t ks = sk*step_size;
      const uint64_t kmax = std::min(ks+step_size,n_samples);
      uint64_t ls = (ks + stripe + 1)%n_samples; // wraparound

      while( ((ks+8) <= kmax) && ((n_samples-ls)>=8) ) {
       UnnormalizedWeighted8<TFloat>(
                                   dm_stripes_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks, ls);
       ks+=8;
       ls = (ls + 8)%n_samples; // wraparound
      } // for ks+=8

      while( ((ks+4) <= kmax) && ((n_samples-ls)>=4) ) {
       UnnormalizedWeighted4<TFloat>(
                                   dm_stripes_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks, ls);
       ks+=4;
       ls = (ls + 4)%n_samples; // wraparound
      } // for ks+=4

      // deal with any leftovers in serial way
      for( ; ks < kmax; ks++ ) {
       UnnormalizedWeighted1<TFloat>(
                                   dm_stripes_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks, ls);
       ls = (ls + 1)%n_samples; // wraparound
      } // for ks
#endif
     } // for stripe
    } // for sk

#ifdef _OPENACC
   // next iteration will use the alternative space
   std::swap(this->embedded_proportions,this->embedded_proportions_alt);
#endif
}

template<class TFloat>
void SUCMP_NM::UnifracVawUnnormalizedWeightedTask<TFloat>::_run(unsigned int filled_embs, const TFloat * __restrict__ lengths) {
    const uint64_t start_idx = this->task_p->start;
    const uint64_t stop_idx = this->task_p->stop;
    const uint64_t n_samples = this->task_p->n_samples;
    const uint64_t n_samples_r = this->dm_stripes.n_samples_r;

    // openacc only works well with local variables
    const TFloat * const __restrict__ embedded_proportions = this->embedded_proportions;
    const TFloat * const __restrict__ embedded_counts = this->embedded_counts;
    const TFloat * const __restrict__ sample_total_counts = this->sample_total_counts;
    TFloat * const __restrict__ dm_stripes_buf = this->dm_stripes.buf;

    const uint64_t step_size = SUCMP_NM::UnifracVawUnnormalizedWeightedTask<TFloat>::step_size;
    const uint64_t sample_steps = (n_samples+(step_size-1))/step_size; // round up

    // point of thread
#ifdef _OPENACC
    const unsigned int acc_vector_size = SUCMP_NM::UnifracVawUnnormalizedWeightedTask<TFloat>::acc_vector_size;
#pragma acc parallel loop collapse(3) vector_length(acc_vector_size) present(embedded_proportions,embedded_counts,sample_total_counts,dm_stripes_buf,lengths) async
#else
#pragma omp parallel for default(shared) schedule(dynamic,1)
#endif
    for(uint64_t sk = 0; sk < sample_steps ; sk++) {
      for(uint64_t stripe = start_idx; stripe < stop_idx; stripe++) {
        for(uint64_t ik = 0; ik < step_size ; ik++) {
            const uint64_t k = sk*step_size + ik;
            const uint64_t idx = (stripe-start_idx) * n_samples_r;
            TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
            //TFloat *dm_stripe = dm_stripes[stripe];

            if (k>=n_samples) continue; // past the limit

            const uint64_t l1 = (k + stripe + 1)%n_samples; // wraparound

            const TFloat m = sample_total_counts[k] + sample_total_counts[l1];

            TFloat my_stripe = dm_stripe[k];

#pragma acc loop seq
            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r*emb;

                TFloat mi = embedded_counts[offset + k] + embedded_counts[offset + l1];
                TFloat vaw = sqrt(mi * (m - mi));

                if(vaw > 0) {
                  TFloat u1 = embedded_proportions[offset + k];
                  TFloat v1 = embedded_proportions[offset + l1];
                  TFloat diff1 = fabs(u1 - v1);
                  TFloat length = lengths[emb];

                   my_stripe += (diff1 * length) / vaw;
                }
            }

            dm_stripe[k]     = my_stripe;
        }

      }
    }

#ifdef _OPENACC
   // next iteration will use the alternative space
   std::swap(this->embedded_proportions,this->embedded_proportions_alt);
#endif
}

// Single step in computing NormalizedWeighted Unifrac
template<class TFloat>
static inline void NormalizedWeighted1(
                      TFloat * const __restrict__ dm_stripes_buf,
                      TFloat * const __restrict__ dm_stripes_total_buf,
                      const bool   * const __restrict__ zcheck,
                      const TFloat * const __restrict__ sums,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t idx,
                      const uint64_t n_samples_r,
                      const uint64_t k,
                      const uint64_t l1) {
       const bool allzero_k = zcheck[k];
       const bool allzero_l1 = zcheck[l1];

       if (allzero_k && allzero_l1) {
         // nothing to do, would have to add 0
       } else {
          TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
          TFloat * const __restrict__ dm_stripe_total = dm_stripes_total_buf+idx;
          //TFloat *dm_stripe = dm_stripes[stripe];
          //TFloat *dm_stripe_total = dm_stripes_total[stripe];

          const TFloat sum_k = sums[k];
          const TFloat sum_l = sums[l1];

          // the totals can always use the distributed property
          dm_stripe_total[k] += sum_k + sum_l;
   
          TFloat my_stripe;

          if (allzero_k || allzero_l1) {
            // one side has all zeros
            // we can use the distributed property, and use the pre-computed values

            my_stripe = (allzero_k) ? sum_l : // if (nonzero_l1) ridx=l1 // fabs(k-l1), with k==0
                                      sum_k;   // if (nonzero_k)  ridx=k  // fabs(k-l1), with l1==0
          } else {
            // both sides non zero, use the explicit but slow approach
            my_stripe = WeightedVal1(embedded_proportions, lengths,
                                     filled_embs, n_samples_r,
                                     k, l1);
          }

          // keep all writes in a single place, to maximize GPU warp performance
          dm_stripe[k]       += my_stripe;
       }
}

#ifndef _OPENACC
// Vectorized step in computing NormalizedWeighted Unifrac
template<class TFloat>
static inline void NormalizedWeighted4(
                      TFloat * const __restrict__ dm_stripes_buf,
                      TFloat * const __restrict__ dm_stripes_total_buf,
                      const bool   * const __restrict__ zcheck,
                      const TFloat * const __restrict__ sums,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t idx,
                      const uint64_t n_samples_r,
                      const uint64_t ks,
                      const uint64_t ls) {
       const uint32_t z_k = ((const uint32_t *)(zcheck+ks))[0];
       const uint32_t z_l = ((const uint32_t *)(zcheck+ls))[0];
       const bool allzero_k = z_k==0x01010101;
       const bool allzero_l = z_l==0x01010101;

       if (allzero_k && allzero_l) {
         // nothing to do, would have to add 0
       } else {
          const TFloat sum_k0 = sums[ks];
          const TFloat sum_k1 = sums[ks+1];
          const TFloat sum_k2 = sums[ks+2];
          const TFloat sum_k3 = sums[ks+3];
          const TFloat sum_l0 = sums[ls];
          const TFloat sum_l1 = sums[ls+1];
          const TFloat sum_l2 = sums[ls+2];
          const TFloat sum_l3 = sums[ls+3];

          const TFloat sum_kl0 = sum_k0 + sum_l0;
          const TFloat sum_kl1 = sum_k1 + sum_l1;
          const TFloat sum_kl2 = sum_k2 + sum_l2;
          const TFloat sum_kl3 = sum_k3 + sum_l3;

          int32_t cnt_k = 0;
          int32_t cnt_l = 0;

          TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
          //TFloat *dm_stripe = dm_stripes[stripe];

          if (allzero_k) {
             // one side has all zeros, use distributed property
             // if (nonzero_l1) ridx=l1 // fabs(k-l1), with k==0
             dm_stripe[ks]   += sum_l0;
             dm_stripe[ks+1] += sum_l1;
             dm_stripe[ks+2] += sum_l2;
             dm_stripe[ks+3] += sum_l3;
          } else if (allzero_l) {
             // one side has all zeros, use distributed property
             // if (nonzero_k)  ridx=k  // fabs(k-l1), with l1==0
             dm_stripe[ks]   += sum_k0;
             dm_stripe[ks+1] += sum_k1;
             dm_stripe[ks+2] += sum_k2;
             dm_stripe[ks+3] += sum_k3;
          } else {
             // popcnt is cheap but has large latency, so compute early
             cnt_k = popcnt_u32(z_k);
             cnt_l = popcnt_u32(z_l);
          }

          // the totals can always use the distributed property
          {
             TFloat * const __restrict__ dm_stripe_total = dm_stripes_total_buf+idx;
             //TFloat *dm_stripe_total = dm_stripes_total[stripe];
             dm_stripe_total[ks]   += sum_kl0;
             dm_stripe_total[ks+1] += sum_kl1;
             dm_stripe_total[ks+2] += sum_kl2;
             dm_stripe_total[ks+3] += sum_kl3;
          }

          if (allzero_k||allzero_l) {
             // already done
          } else if ((cnt_k<3) && (cnt_l<3)) {
             // several of the elemens are nonzero, may as well use the vectorized version
             TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
             //TFloat *dm_stripe = dm_stripes[stripe];

             WeightedVal4(dm_stripe,
                       embedded_proportions, lengths,
                       filled_embs, n_samples_r,
                       ks, ls);
          } else {
             // only a few have both sides partially non zero, use the fine grained compute
             // Use UnnormalizedWeighted since we already computed dm_stripe_total
             for (uint64_t i=0; i<4; i++) {
                UnnormalizedWeighted1<TFloat>(
                                   dm_stripes_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks+i, ls+i);
             }
          }
       } // (allzero_k && allzero_l)
}

template<class TFloat>
static inline void NormalizedWeighted8(
                      TFloat * const __restrict__ dm_stripes_buf,
                      TFloat * const __restrict__ dm_stripes_total_buf,
                      const bool   * const __restrict__ zcheck,
                      const TFloat * const __restrict__ sums,
                      const TFloat * const __restrict__ embedded_proportions,
                      const TFloat * __restrict__ lengths,
                      const unsigned int filled_embs,
                      const uint64_t idx,
                      const uint64_t n_samples_r,
                      const uint64_t ks,
                      const uint64_t ls) {
       const uint64_t z_k = ((const uint64_t *)(zcheck+ks))[0];
       const uint64_t z_l = ((const uint64_t *)(zcheck+ls))[0];
       const bool allzero_k = z_k==0x0101010101010101;
       const bool allzero_l = z_l==0x0101010101010101;

       if (allzero_k && allzero_l) {
         // nothing to do, would have to add 0
       } else {
          const TFloat sum_k0 = sums[ks];
          const TFloat sum_k1 = sums[ks+1];
          const TFloat sum_k2 = sums[ks+2];
          const TFloat sum_k3 = sums[ks+3];
          const TFloat sum_k4 = sums[ks+4];
          const TFloat sum_k5 = sums[ks+5];
          const TFloat sum_k6 = sums[ks+6];
          const TFloat sum_k7 = sums[ks+7];
          const TFloat sum_l0 = sums[ls];
          const TFloat sum_l1 = sums[ls+1];
          const TFloat sum_l2 = sums[ls+2];
          const TFloat sum_l3 = sums[ls+3];
          const TFloat sum_l4 = sums[ls+4];
          const TFloat sum_l5 = sums[ls+5];
          const TFloat sum_l6 = sums[ls+6];
          const TFloat sum_l7 = sums[ls+7];

          const TFloat sum_kl0 = sum_k0 + sum_l0;
          const TFloat sum_kl1 = sum_k1 + sum_l1;
          const TFloat sum_kl2 = sum_k2 + sum_l2;
          const TFloat sum_kl3 = sum_k3 + sum_l3;
          const TFloat sum_kl4 = sum_k4 + sum_l4;
          const TFloat sum_kl5 = sum_k5 + sum_l5;
          const TFloat sum_kl6 = sum_k6 + sum_l6;
          const TFloat sum_kl7 = sum_k7 + sum_l7;

          int32_t cnt_k = 0;
          int32_t cnt_l = 0;

          TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
          //TFloat *dm_stripe = dm_stripes[stripe];

          if (allzero_k) {
             // one side has all zeros, use distributed property
             // if (nonzero_l1) ridx=l1 // fabs(k-l1), with k==0
             dm_stripe[ks]   += sum_l0;
             dm_stripe[ks+1] += sum_l1;
             dm_stripe[ks+2] += sum_l2;
             dm_stripe[ks+3] += sum_l3;
             dm_stripe[ks+4] += sum_l4;
             dm_stripe[ks+5] += sum_l5;
             dm_stripe[ks+6] += sum_l6;
             dm_stripe[ks+7] += sum_l7;
          } else if (allzero_l) {
             // one side has all zeros, use distributed property
             // if (nonzero_k)  ridx=k  // fabs(k-l1), with l1==0
             dm_stripe[ks]   += sum_k0;
             dm_stripe[ks+1] += sum_k1;
             dm_stripe[ks+2] += sum_k2;
             dm_stripe[ks+3] += sum_k3;
             dm_stripe[ks+4] += sum_k4;
             dm_stripe[ks+5] += sum_k5;
             dm_stripe[ks+6] += sum_k6;
             dm_stripe[ks+7] += sum_k7;
          } else {
             // popcnt is cheap but has large latency, so compute early
             cnt_k = popcnt_u64(z_k);
             cnt_l = popcnt_u64(z_l);
          }

          // the totals can always use the distributed property
          {
             TFloat * const __restrict__ dm_stripe_total = dm_stripes_total_buf+idx;
             //TFloat *dm_stripe_total = dm_stripes_total[stripe];
             dm_stripe_total[ks]   += sum_kl0;
             dm_stripe_total[ks+1] += sum_kl1;
             dm_stripe_total[ks+2] += sum_kl2;
             dm_stripe_total[ks+3] += sum_kl3;
             dm_stripe_total[ks+4] += sum_kl4;
             dm_stripe_total[ks+5] += sum_kl5;
             dm_stripe_total[ks+6] += sum_kl6;
             dm_stripe_total[ks+7] += sum_kl7;
          }

          if (allzero_k||allzero_l) {
             // already done
          } else if ((cnt_k<6) && (cnt_l<6)) {
             // several of the elemens are nonzero, may as well use the vectorized version
             TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
             //TFloat *dm_stripe = dm_stripes[stripe];

             WeightedVal8(dm_stripe,
                       embedded_proportions, lengths,
                       filled_embs, n_samples_r,
                       ks, ls);
          } else {
             // only a few have both sides partially non zero, use the fine grained compute
             // Use UnnormalizedWeighted since we already computed dm_stripe_total
             for (uint64_t i=0; i<8; i++) {
                UnnormalizedWeighted1<TFloat>(
                                   dm_stripes_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks+i, ls+i);
             }
          }
       } // (allzero_k && allzero_l)
}
#endif

template<class TFloat>
void SUCMP_NM::UnifracNormalizedWeightedTask<TFloat>::_run(unsigned int filled_embs, const TFloat * __restrict__ lengths) {
    const uint64_t start_idx = this->task_p->start;
    const uint64_t stop_idx = this->task_p->stop;
    const uint64_t n_samples = this->task_p->n_samples;
    const uint64_t n_samples_r = this->dm_stripes.n_samples_r;

    // openacc only works well with local variables
    const TFloat * const __restrict__ embedded_proportions = this->embedded_proportions;
    TFloat * const __restrict__ dm_stripes_buf = this->dm_stripes.buf;
    TFloat * const __restrict__ dm_stripes_total_buf = this->dm_stripes_total.buf;

    bool * const __restrict__ zcheck = this->zcheck;
    TFloat * const __restrict__ sums = this->sums;

    const uint64_t step_size = SUCMP_NM::UnifracNormalizedWeightedTask<TFloat>::step_size;
    const uint64_t sample_steps = (n_samples+(step_size-1))/step_size; // round up

    // check for zero values and pre-compute single column sums
    WeightedZerosAndSums(zcheck, sums,
                         embedded_proportions, lengths,
                         filled_embs, n_samples, n_samples_r);

    // point of thread
#ifdef _OPENACC
    const unsigned int acc_vector_size = SUCMP_NM::UnifracNormalizedWeightedTask<TFloat>::acc_vector_size;
#pragma acc parallel loop gang vector collapse(3) vector_length(acc_vector_size) present(embedded_proportions,dm_stripes_buf,dm_stripes_total_buf,lengths,zcheck,sums) async
#else
    // use dynamic scheduling due to non-homogeneity in the loop
#pragma omp parallel for schedule(dynamic,1) default(shared)
#endif
    for(uint64_t sk = 0; sk < sample_steps ; sk++) {
     for(uint64_t stripe = start_idx; stripe < stop_idx; stripe++) {
#ifdef _OPENACC
      // SIMT-based GPU work great one at a time (HW will deal with parallelism)
      for(uint64_t ik = 0; ik < step_size ; ik++) {
       const uint64_t k = sk*step_size + ik;

       if (k>=n_samples) continue; // past the limit

       const uint64_t l1 = (k + stripe + 1)%n_samples; // wraparound
       const uint64_t idx = (stripe-start_idx) * n_samples_r;

       NormalizedWeighted1<TFloat>(
                                   dm_stripes_buf,dm_stripes_total_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   k, l1);
      } // for ik
#else
      // SIMD-based CPUs need help with vectorization
      const uint64_t idx = (stripe-start_idx) * n_samples_r;
      uint64_t ks = sk*step_size;
      const uint64_t kmax = std::min(ks+step_size,n_samples);
      uint64_t ls = (ks + stripe + 1)%n_samples; // wraparound

      while( ((ks+8) <= kmax) && ((n_samples-ls)>=8) ) {
       NormalizedWeighted8<TFloat>(
                                   dm_stripes_buf,dm_stripes_total_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks, ls);
       ks+=8;
       ls = (ls + 8)%n_samples; // wraparound
      } // for ks+=8

      while( ((ks+4) <= kmax) && ((n_samples-ls)>=4) ) {
       NormalizedWeighted4<TFloat>(
                                   dm_stripes_buf,dm_stripes_total_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks, ls);
       ks+=4;
       ls = (ls + 4)%n_samples; // wraparound
      } // for ks+=4

      // deal with any leftovers in serial way
      for( ; ks < kmax; ks++ ) {
       NormalizedWeighted1<TFloat>(
                                   dm_stripes_buf,dm_stripes_total_buf,
                                   zcheck, sums, embedded_proportions, lengths,
                                   filled_embs,idx, n_samples_r,
                                   ks, ls);
       ls = (ls + 1)%n_samples; // wraparound
      } // for ks
#endif
     } // for stripe
    } // for sk

#ifdef _OPENACC
   // next iteration will use the alternative space
   std::swap(this->embedded_proportions,this->embedded_proportions_alt);
#endif
}

template<class TFloat>
void SUCMP_NM::UnifracVawNormalizedWeightedTask<TFloat>::_run(unsigned int filled_embs, const TFloat * __restrict__ lengths) {
    const uint64_t start_idx = this->task_p->start;
    const uint64_t stop_idx = this->task_p->stop;
    const uint64_t n_samples = this->task_p->n_samples;
    const uint64_t n_samples_r = this->dm_stripes.n_samples_r;

    // openacc only works well with local variables
    const TFloat * const __restrict__ embedded_proportions = this->embedded_proportions;
    const TFloat * const __restrict__ embedded_counts = this->embedded_counts;
    const TFloat * const __restrict__ sample_total_counts = this->sample_total_counts;
    TFloat * const __restrict__ dm_stripes_buf = this->dm_stripes.buf;
    TFloat * const __restrict__ dm_stripes_total_buf = this->dm_stripes_total.buf;

    const uint64_t step_size = SUCMP_NM::UnifracVawNormalizedWeightedTask<TFloat>::step_size;
    const uint64_t sample_steps = (n_samples+(step_size-1))/step_size; // round up

    // point of thread
#ifdef _OPENACC
    const unsigned int acc_vector_size = SUCMP_NM::UnifracVawNormalizedWeightedTask<TFloat>::acc_vector_size;
#pragma acc parallel loop collapse(3) vector_length(acc_vector_size) present(embedded_proportions,embedded_counts,sample_total_counts,dm_stripes_buf,dm_stripes_total_buf,lengths) async
#else
#pragma omp parallel for schedule(dynamic,1) default(shared)
#endif
    for(uint64_t sk = 0; sk < sample_steps ; sk++) {
      for(uint64_t stripe = start_idx; stripe < stop_idx; stripe++) {
        for(uint64_t ik = 0; ik < step_size ; ik++) {
            const uint64_t k = sk*step_size + ik;
            const uint64_t idx = (stripe-start_idx) * n_samples_r;
            TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
            TFloat * const __restrict__ dm_stripe_total = dm_stripes_total_buf+idx;
            //TFloat *dm_stripe = dm_stripes[stripe];
            //TFloat *dm_stripe_total = dm_stripes_total[stripe];

            if (k>=n_samples) continue; // past the limit

            const uint64_t l1 = (k + stripe + 1)%n_samples; // wraparound

            const TFloat m = sample_total_counts[k] + sample_total_counts[l1];

            TFloat my_stripe = dm_stripe[k];
            TFloat my_stripe_total = dm_stripe_total[k];

#pragma acc loop seq
            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r * emb;

                TFloat mi = embedded_counts[offset + k] + embedded_counts[offset + l1];
                TFloat vaw = sqrt(mi * (m - mi));

                if(vaw > 0) {
                  TFloat u1 = embedded_proportions[offset + k];
                  TFloat v1 = embedded_proportions[offset + l1];
                  TFloat diff1 = fabs(u1 - v1);
                  TFloat length = lengths[emb];

                  my_stripe += (diff1 * length) / vaw;
                  my_stripe_total += ((u1 + v1) * length) / vaw;
                }
            }

            dm_stripe[k]     = my_stripe;
            dm_stripe_total[k]     = my_stripe_total;

        }

      }
    }

#ifdef _OPENACC
   // next iteration will use the alternative space
   std::swap(this->embedded_proportions,this->embedded_proportions_alt);
#endif
}

template<class TFloat>
void SUCMP_NM::UnifracGeneralizedTask<TFloat>::_run(unsigned int filled_embs, const TFloat * __restrict__ lengths) {
    const uint64_t start_idx = this->task_p->start;
    const uint64_t stop_idx = this->task_p->stop;
    const uint64_t n_samples = this->task_p->n_samples;
    const uint64_t n_samples_r = this->dm_stripes.n_samples_r;

    // openacc only works well with local variables
    const TFloat * const __restrict__ embedded_proportions = this->embedded_proportions;
    TFloat * const __restrict__ dm_stripes_buf = this->dm_stripes.buf;
    TFloat * const __restrict__ dm_stripes_total_buf = this->dm_stripes_total.buf;

    const TFloat g_unifrac_alpha = this->task_p->g_unifrac_alpha;

    const uint64_t step_size = SUCMP_NM::UnifracGeneralizedTask<TFloat>::step_size;
    const uint64_t sample_steps = (n_samples+(step_size-1))/step_size; // round up

    // point of thread
#ifdef _OPENACC
    const unsigned int acc_vector_size = SUCMP_NM::UnifracGeneralizedTask<TFloat>::acc_vector_size;
#pragma acc parallel loop collapse(3) vector_length(acc_vector_size) present(embedded_proportions,dm_stripes_buf,dm_stripes_total_buf,lengths) async
#else
#pragma omp parallel for schedule(dynamic,1) default(shared)
#endif
    for(uint64_t sk = 0; sk < sample_steps ; sk++) {
      for(uint64_t stripe = start_idx; stripe < stop_idx; stripe++) {
        for(uint64_t ik = 0; ik < step_size ; ik++) {
            const uint64_t k = sk*step_size + ik;
            const uint64_t idx = (stripe-start_idx) * n_samples_r;
            TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
            TFloat * const __restrict__ dm_stripe_total = dm_stripes_total_buf+idx;
            //TFloat *dm_stripe = dm_stripes[stripe];
            //TFloat *dm_stripe_total = dm_stripes_total[stripe];

            if (k>=n_samples) continue; // past the limit

            const uint64_t l1 = (k + stripe + 1)%n_samples; // wraparound

            TFloat my_stripe = dm_stripe[k];
            TFloat my_stripe_total = dm_stripe_total[k];

#pragma acc loop seq
            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r * emb;

                TFloat u1 = embedded_proportions[offset + k];
                TFloat v1 = embedded_proportions[offset + l1];
                TFloat sum1 = u1 + v1;

                if(sum1 != 0.0) { 
                   TFloat length = lengths[emb];
                   TFloat diff1 = fabs(u1 - v1);
                   TFloat sum_pow1 = pow(sum1, g_unifrac_alpha) * length; 

                   my_stripe += sum_pow1 * (diff1 / sum1); 
                   my_stripe_total += sum_pow1; 
                }
            }

            dm_stripe[k]     = my_stripe;
            dm_stripe_total[k]     = my_stripe_total;

        }

      }
    }

#ifdef _OPENACC
   // next iteration will use the alternative space
   std::swap(this->embedded_proportions,this->embedded_proportions_alt);
#endif
}

template<class TFloat>
void SUCMP_NM::UnifracVawGeneralizedTask<TFloat>::_run(unsigned int filled_embs, const TFloat * __restrict__ lengths) {
    const uint64_t start_idx = this->task_p->start;
    const uint64_t stop_idx = this->task_p->stop;
    const uint64_t n_samples = this->task_p->n_samples;
    const uint64_t n_samples_r = this->dm_stripes.n_samples_r;

    const TFloat g_unifrac_alpha = this->task_p->g_unifrac_alpha;

    // openacc only works well with local variables
    const TFloat * const __restrict__ embedded_proportions = this->embedded_proportions;
    const TFloat * const __restrict__ embedded_counts = this->embedded_counts;
    const TFloat * const __restrict__ sample_total_counts = this->sample_total_counts;
    TFloat * const __restrict__ dm_stripes_buf = this->dm_stripes.buf;
    TFloat * const __restrict__ dm_stripes_total_buf = this->dm_stripes_total.buf;

    const uint64_t step_size = SUCMP_NM::UnifracVawGeneralizedTask<TFloat>::step_size;
    const uint64_t sample_steps = (n_samples+(step_size-1))/step_size; // round up
    // quick hack, to be finished

    // point of thread
#ifdef _OPENACC
    const unsigned int acc_vector_size = SUCMP_NM::UnifracVawGeneralizedTask<TFloat>::acc_vector_size;
#pragma acc parallel loop collapse(3) vector_length(acc_vector_size) present(embedded_proportions,embedded_counts,sample_total_counts,dm_stripes_buf,dm_stripes_total_buf,lengths) async
#else
#pragma omp parallel for schedule(dynamic,1) default(shared)
#endif
    for(uint64_t sk = 0; sk < sample_steps ; sk++) {
      for(uint64_t stripe = start_idx; stripe < stop_idx; stripe++) {
        for(uint64_t ik = 0; ik < step_size ; ik++) {
            const uint64_t k = sk*step_size + ik;
            const uint64_t idx = (stripe-start_idx) * n_samples_r;
            TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
            TFloat * const __restrict__ dm_stripe_total = dm_stripes_total_buf+idx;
            //TFloat *dm_stripe = dm_stripes[stripe];
            //TFloat *dm_stripe_total = dm_stripes_total[stripe];

            if (k>=n_samples) continue; // past the limit

            const uint64_t l1 = (k + stripe + 1)%n_samples; // wraparound

            const TFloat m = sample_total_counts[k] + sample_total_counts[l1];

            TFloat my_stripe = dm_stripe[k];
            TFloat my_stripe_total = dm_stripe_total[k];

#pragma acc loop seq
            for (uint64_t emb=0; emb<filled_embs; emb++) {
                const uint64_t offset = n_samples_r * emb;

                TFloat mi = embedded_counts[offset + k] + embedded_counts[offset + l1];
                TFloat vaw = sqrt(mi * (m - mi));

                if(vaw > 0) {
                  TFloat u1 = embedded_proportions[offset + k];
                  TFloat v1 = embedded_proportions[offset + l1];
                  TFloat length = lengths[emb];

                  TFloat sum1 = (u1 + v1) / vaw;
                  TFloat sub1 = fabs(u1 - v1) / vaw;
                  TFloat sum_pow1 = pow(sum1, g_unifrac_alpha) * length;

                  my_stripe += sum_pow1 * (sub1 / sum1);
                  my_stripe_total += sum_pow1;
                }
            }

            dm_stripe[k]     = my_stripe;
            dm_stripe_total[k]     = my_stripe_total;

        }
      }
    }

#ifdef _OPENACC
   // next iteration will use the alternative space
   std::swap(this->embedded_proportions,this->embedded_proportions_alt);
#endif
}

template<class TFloat>
void SUCMP_NM::UnifracUnweightedTask<TFloat>::_run(unsigned int filled_embs, const TFloat * __restrict__ lengths) {
    const uint64_t start_idx = this->task_p->start;
    const uint64_t stop_idx = this->task_p->stop;
    const uint64_t n_samples = this->task_p->n_samples;
    const uint64_t n_samples_r = this->dm_stripes.n_samples_r;

    // openacc only works well with local variables
    const uint64_t * const __restrict__ embedded_proportions = this->embedded_proportions;
    TFloat * const __restrict__ dm_stripes_buf = this->dm_stripes.buf;
    TFloat * const __restrict__ dm_stripes_total_buf = this->dm_stripes_total.buf;

    bool * const __restrict__ zcheck = this->zcheck;
    TFloat * const __restrict__ sums = this->sums;

    const uint64_t step_size = SUCMP_NM::UnifracUnweightedTask<TFloat>::step_size;
    const uint64_t sample_steps = (n_samples+(step_size-1))/step_size; // round up

    const uint64_t filled_embs_els = filled_embs/64;
    const uint64_t filled_embs_rem = filled_embs%64; 

    const uint64_t filled_embs_els_round = (filled_embs+63)/64;

    // check for zero values 
    WeightedZerosAsync(zcheck,
                       embedded_proportions,
                       filled_embs_els_round, n_samples, n_samples_r);


    // pre-compute sums of length elements, since they are likely to be accessed many times
    // We will use a 8-bit map, to keep it small enough to keep in L1 cache
#ifdef _OPENACC
#pragma acc parallel loop collapse(2) gang present(lengths,sums) async
#else 
#pragma omp parallel for default(shared)
#endif
    for (uint64_t emb_el=0; emb_el<filled_embs_els; emb_el++) {
       for (uint64_t sub8=0; sub8<8; sub8++) {
          const uint64_t emb8 = emb_el*8+sub8;
          TFloat * __restrict__ psum = &(sums[emb8<<8]);
          const TFloat * __restrict__ pl   = &(lengths[emb8*8]);

#pragma acc loop vector
          // compute all the combinations for this block (8-bits total)
          // psum[0] = 0.0   // +0*pl[0]+0*pl[1]+0*pl[2]+...
          // psum[1] = pl[0] // +0*pl[1]+0*pl[2]+...
          // psum[2] = pl[1] // +0*pl[0]+0*pl[2]+
          // psum[2] = pl[0] + pl[1]
          // ...
          // psum[255] = pl[1] +.. + pl[7] // + 0*pl[0]
          // psum[255] = pl[0] +pl[1] +.. + pl[7]
          for (uint64_t b8_i=0; b8_i<0x100; b8_i++) {
             psum[b8_i] = (((b8_i >> 0) & 1) * pl[0]) + (((b8_i >> 1) & 1) * pl[1]) + 
                          (((b8_i >> 2) & 1) * pl[2]) + (((b8_i >> 3) & 1) * pl[3]) +
                          (((b8_i >> 4) & 1) * pl[4]) + (((b8_i >> 5) & 1) * pl[5]) +
                          (((b8_i >> 6) & 1) * pl[6]) + (((b8_i >> 7) & 1) * pl[7]);
          }
       }
    }
    if (filled_embs_rem>0) { // add also the overflow elements
       const uint64_t emb_el=filled_embs_els;
#ifdef _OPENACC
#pragma acc parallel loop gang present(lengths,sums) async
#else
       // no advantage of OMP, too small
#endif
       for (uint64_t sub8=0; sub8<8; sub8++) {
          // we are summing we have enough buffer in sums
          const uint64_t emb8 = emb_el*8+sub8;
          TFloat * __restrict__ psum = &(sums[emb8<<8]);

#pragma acc loop vector
          // compute all the combinations for this block, set to 0 any past the limit
          // as above
          for (uint64_t b8_i=0; b8_i<0x100; b8_i++) {
             TFloat val= 0;
             for (uint64_t li=(emb8*8); li<filled_embs; li++) {
               val += ((b8_i >>  (li-(emb8*8))) & 1) * lengths[li];
             }
             psum[b8_i] = val;
          }
        }
    }

    // point of thread
#ifdef _OPENACC
#pragma acc wait
    const unsigned int acc_vector_size = SUCMP_NM::UnifracUnweightedTask<TFloat>::acc_vector_size;
#pragma acc parallel loop collapse(3) vector_length(acc_vector_size) present(embedded_proportions,dm_stripes_buf,dm_stripes_total_buf,sums,zcheck) async
#else
    // use dynamic scheduling due to non-homogeneity in the loop
#pragma omp parallel for schedule(dynamic,1) default(shared)
#endif
    for(uint64_t sk = 0; sk < sample_steps ; sk++) {
      for(uint64_t stripe = start_idx; stripe < stop_idx; stripe++) {
        for(uint64_t ik = 0; ik < step_size ; ik++) {
            const uint64_t k = sk*step_size + ik;
            const uint64_t idx = (stripe-start_idx) * n_samples_r;
            TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
            TFloat * const __restrict__ dm_stripe_total = dm_stripes_total_buf+idx;
            //TFloat *dm_stripe = dm_stripes[stripe];
            //TFloat *dm_stripe_total = dm_stripes_total[stripe];

            if (k>=n_samples) continue; // past the limit

            const uint64_t l1 = (k + stripe + 1)%n_samples; // wraparound

            const bool allzero_k = zcheck[k];
            const bool allzero_l1 = zcheck[l1];

            if (allzero_k && allzero_l1) {
             // nothing to do, would have to add 0
            } else {
             bool did_update = false;
             TFloat my_stripe = 0.0;
             TFloat my_stripe_total = 0.0;

             if (allzero_k || allzero_l1) {
              // with one side zero, | and ^ are no-ops
              const uint64_t kl = (allzero_k) ? l1 : k; // only use the non-sero one
#pragma acc loop seq
              for (uint64_t emb_el=0; emb_el<filled_embs_els_round; emb_el++) {
                const uint64_t offset = n_samples_r * emb_el;
                const TFloat * __restrict__ psum = &(sums[emb_el*0x800]);

                uint64_t o1 = embedded_proportions[offset + kl];

                if (o1!=0) {  // zeros are prevalent
                    did_update=true;
                    uint64_t x1 = o1; // just for consistency 

                    // Use the pre-computed sums
                    // Each range of 8 lengths has already been pre-computed and stored in psum
                    // Since embedded_proportions packed format is in 64-bit format for performance reasons
                    //    we need to add the 8 sums using the four 8-bits for addressing inside psum

                    const TFloat my_stripe_sum = 
                                   psum[              (x1 & 0xff)] + 
                                   psum[0x100+((x1 >>  8) & 0xff)] +
                                   psum[0x200+((x1 >> 16) & 0xff)] +
                                   psum[0x300+((x1 >> 24) & 0xff)] +
                                   psum[0x400+((x1 >> 32) & 0xff)] +
                                   psum[0x500+((x1 >> 40) & 0xff)] +
                                   psum[0x600+((x1 >> 48) & 0xff)] +
                                   psum[0x700+((x1 >> 56)       )];
                   my_stripe       += my_stripe_sum;
                   my_stripe_total += my_stripe_sum;
                }
              } // for emb_el
             } else {
              // we need both sides
#pragma acc loop seq
              for (uint64_t emb_el=0; emb_el<filled_embs_els_round; emb_el++) {
                const uint64_t offset = n_samples_r * emb_el;
                const TFloat * __restrict__ psum = &(sums[emb_el*0x800]);

                uint64_t u1 = embedded_proportions[offset + k];
                uint64_t v1 = embedded_proportions[offset + l1];
                uint64_t o1 = u1 | v1;

                if (o1!=0) {  // zeros are prevalent
                    did_update=true;
                    uint64_t x1 = u1 ^ v1;

                    // Use the pre-computed sums
                    // Each range of 8 lengths has already been pre-computed and stored in psum
                    // Since embedded_proportions packed format is in 64-bit format for performance reasons
                    //    we need to add the 8 sums using the four 8-bits for addressing inside psum

                    my_stripe       += psum[              (x1 & 0xff)] + 
                                   psum[0x100+((x1 >>  8) & 0xff)] +
                                   psum[0x200+((x1 >> 16) & 0xff)] +
                                   psum[0x300+((x1 >> 24) & 0xff)] +
                                   psum[0x400+((x1 >> 32) & 0xff)] +
                                   psum[0x500+((x1 >> 40) & 0xff)] +
                                   psum[0x600+((x1 >> 48) & 0xff)] +
                                   psum[0x700+((x1 >> 56)       )];
                    my_stripe_total += psum[              (o1 & 0xff)] +
                                   psum[0x100+((o1 >>  8) & 0xff)] +
                                   psum[0x200+((o1 >> 16) & 0xff)] +
                                   psum[0x300+((o1 >> 24) & 0xff)] +
                                   psum[0x400+((o1 >> 32) & 0xff)] +
                                   psum[0x500+((o1 >> 40) & 0xff)] +
                                   psum[0x600+((o1 >> 48) & 0xff)] +
                                   psum[0x700+((o1 >> 56)       )];
                }
              } // for emb_el
             }

             if (did_update) {
               dm_stripe[k]       += my_stripe;
               dm_stripe_total[k] += my_stripe_total;
             }
            } // if (allzero_k && allzero_l1)
        } // for ik

      }
    }

#ifdef _OPENACC
   // next iteration will use the alternative space
   std::swap(this->embedded_proportions,this->embedded_proportions_alt);
#endif
}

template<class TFloat>
void SUCMP_NM::UnifracVawUnweightedTask<TFloat>::_run(unsigned int filled_embs, const TFloat * __restrict__ lengths) {
    const uint64_t start_idx = this->task_p->start;
    const uint64_t stop_idx = this->task_p->stop;
    const uint64_t n_samples = this->task_p->n_samples;
    const uint64_t n_samples_r = this->dm_stripes.n_samples_r;

    // openacc only works well with local variables
    const uint32_t * const __restrict__ embedded_proportions = this->embedded_proportions;
    const TFloat  * const __restrict__ embedded_counts = this->embedded_counts;
    const TFloat  * const __restrict__ sample_total_counts = this->sample_total_counts;
    TFloat * const __restrict__ dm_stripes_buf = this->dm_stripes.buf;
    TFloat * const __restrict__ dm_stripes_total_buf = this->dm_stripes_total.buf;

    const uint64_t step_size = SUCMP_NM::UnifracVawUnweightedTask<TFloat>::step_size;
    const uint64_t sample_steps = (n_samples+(step_size-1))/step_size; // round up

    const uint64_t filled_embs_els = (filled_embs+31)/32; // round up

    // point of thread
#ifdef _OPENACC
    const unsigned int acc_vector_size = SUCMP_NM::UnifracVawUnweightedTask<TFloat>::acc_vector_size;
#pragma acc parallel loop collapse(3) vector_length(acc_vector_size) present(embedded_proportions,embedded_counts,sample_total_counts,dm_stripes_buf,dm_stripes_total_buf,lengths) async
#else
#pragma omp parallel for schedule(dynamic,1) default(shared)
#endif
    for(uint64_t sk = 0; sk < sample_steps ; sk++) {
      for(uint64_t stripe = start_idx; stripe < stop_idx; stripe++) {
        for(uint64_t ik = 0; ik < step_size ; ik++) {
            const uint64_t k = sk*step_size + ik;
            const uint64_t idx = (stripe-start_idx) * n_samples_r;
            TFloat * const __restrict__ dm_stripe = dm_stripes_buf+idx;
            TFloat * const __restrict__ dm_stripe_total = dm_stripes_total_buf+idx;
            //TFloat *dm_stripe = dm_stripes[stripe];
            //TFloat *dm_stripe_total = dm_stripes_total[stripe];

            if (k>=n_samples) continue; // past the limit

            const uint64_t l1 = (k + stripe + 1)%n_samples; // wraparound

            TFloat my_stripe = dm_stripe[k];
            TFloat my_stripe_total = dm_stripe_total[k];

            const TFloat m = sample_total_counts[k] + sample_total_counts[l1];

#pragma acc loop seq
            for (uint64_t emb_el=0; emb_el<filled_embs_els; emb_el++) {
                const uint64_t offset_p = n_samples_r * emb_el;
                uint32_t u1 = embedded_proportions[offset_p + k];
                uint32_t v1 = embedded_proportions[offset_p + l1];
                uint32_t x1 = u1 ^ v1;
                uint32_t o1 = u1 | v1;

                // embedded_proporions is packed

#pragma acc loop seq
                for (uint64_t ei=0; ei<32; ei++) {
                   uint64_t emb=emb_el*32+ei;
                   if (emb<filled_embs) {
                     const uint64_t offset_c = n_samples_r * emb;
                     TFloat mi = embedded_counts[offset_c + k] + embedded_counts[offset_c + l1];
                     TFloat vaw = sqrt(mi * (m - mi));

                     // embedded_counts is not packed

                     if(vaw > 0) {
                       TFloat length = lengths[emb];
                       TFloat lv1 = length / vaw;

                       my_stripe +=       ((x1 >> ei) & 1)*lv1;
                       my_stripe_total += ((o1 >> ei) & 1)*lv1;
                     }
                   }
                }
            }

            dm_stripe[k]     = my_stripe;
            dm_stripe_total[k]     = my_stripe_total;

        }

      }
    }

#ifdef _OPENACC
   // next iteration will use the alternative space
   std::swap(this->embedded_proportions,this->embedded_proportions_alt);
#endif
}

