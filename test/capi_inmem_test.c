#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "api.hpp"

#ifndef bool
#define bool char
#define true 1
#define false 0
#endif

void err(bool condition, const char* msg) {
    if(condition) {
        fprintf(stderr, "%s\n", msg);
        exit(1);
    }
} 

// biom
const unsigned int n_obs = 5;
const unsigned int n_samp = 6;
const char* const obs_ids[] = {"GG_OTU_1", "GG_OTU_2", "GG_OTU_3", "GG_OTU_4", "GG_OTU_5"};
const char* const samp_ids[] = {"Sample1", "Sample2", "Sample3", "Sample4", "Sample5", "Sample6"};
const uint32_t    indices[] = {2, 0, 1, 3, 4, 5, 2, 3, 5, 0, 1, 2, 5, 1, 2};
const uint32_t    indptr[] = {0,  1,  6,  9, 13, 15};
const double      data[]= {1., 5., 1., 2., 3., 1., 1., 4., 2., 2., 1., 1., 1., 1., 1.};

// tree
const unsigned int nparens = 16;
const double lengths[] = { 0, 0, 0, 0,
	                   0, 0, 0, 0,
	                   0, 0, 0, 0,
	                   0, 0, 0, 0 };
const bool structure[] = { true, true, false, true,
	                   true, false, true, false,
			   false, true, true, false,
			   true, false, false, false };
const char * const names[] = {"", "GG_OTU_1", "", "",
                              "GG_OTU_2", "", "GG_OTU_3", "",
			      "", "", "GG_OTU_5", "",
			      "GG_OTU_4", "", "", ""};

void test_su_dense(int num_cores){
    double result = 0.0;
    const support_bptree_t tree = {(bool*) structure, (double*) lengths, (char**) names, nparens};
    const char* table_oids[] = { "GG_OTU_1", "GG_OTU_2", "GG_OTU_3", "GG_OTU_4", "GG_OTU_5" };
    const double sample1[] = { 0, 5, 0, 2, 0 };
    const double sample2[] = { 0, 1, 0, 1, 1 };
    const double sample3[] = { 1, 0, 1, 1, 1 };
    const char* method = "unweighted";
    float exp13 = 0.57142857;
    float exp21 = 0.2;
    
    opaque_bptree_t *tree_data;
    convert_bptree_opaque(&tree, &tree_data);

    ComputeStatus status;
    status = one_dense_pair_v2t(5, table_oids, sample1, sample3,
	                        tree_data, method,
                                false, 1.0, false,
				&result);

    err(status != okay, "Compute failed");

    err(fabs(exp13 - result) > 0.00001, "Result is wrong");

    status = one_dense_pair_v2t(5, table_oids, sample2, sample1,
	                        tree_data, method,
                                false, 1.0, false,
				&result);

    err(status != okay, "Compute failed");

    err(fabs(exp21 - result) > 0.00001, "Result is wrong");

    destroy_bptree_opaque(&tree_data);

    // check direct, expanded tree structure, too
    status = one_dense_pair_v2(5, table_oids, sample1, sample3,
	                       &tree, method,
                               false, 1.0, false,
			       &result);

    err(status != okay, "Compute failed");

    err(fabs(exp13 - result) > 0.00001, "Result is wrong");

}

void test_su_matrix(int num_cores){
    mat_full_fp64_t* result = NULL;
    mat_full_fp32_t* result_fp32 = NULL;
    mat_full_fp64_t* result2 = NULL;
    mat_full_fp32_t* result2_fp32 = NULL;
    const support_biom_t table = {(char**) obs_ids, (char**) samp_ids, (uint32_t*) indices, (uint32_t*) indptr, (double*) data, n_obs, n_samp, 0};
    const support_bptree_t tree = {(bool*) structure, (double*) lengths, (char**) names, nparens};

    float exp[] = { 0.0,        0.2,        0.57142857, 0.6,        0.5,        0.2, 
                    0.2,        0.0,        0.42857143, 0.66666667, 0.6,        0.33333333,
		    0.57142857, 0.42857143, 0.0,        0.71428571, 0.85714286, 0.42857143, 
		    0.6,        0.66666667, 0.71428571, 0.0,        0.33333333, 0.4,
		    0.5,        0.6,        0.85714286, 0.33333333, 0.0,        0.6,
                    0.2,        0.33333333, 0.42857143, 0.4,        0.6,        0.0        };
    
    ComputeStatus status;

    status = one_off_matrix_inmem_v2(&table, &tree, "unweighted_fp64",
                                    false, 1.0, false, num_cores,
                                    0, true, NULL,
				    &result);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result->n_samples != 6, "Wrong number of samples");

    for(unsigned int i = 0; i < (result->n_samples*result->n_samples); i++) {
        err(fabs(exp[i] - result->matrix[i]) > 0.00001, "Result is wrong");
    }


    destroy_mat_full_fp64(&result);

    status = one_off_matrix_inmem_fp32_v2(&table, &tree, "unweighted_fp32",
                                    false, 1.0, false, num_cores,
                                    0, true, NULL,
				    &result_fp32);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result_fp32->n_samples != 6, "Wrong number of samples");

    for(unsigned int i = 0; i < (result_fp32->n_samples*result_fp32->n_samples); i++) {
        err(fabs(exp[i] - result_fp32->matrix[i]) > 0.00001, "Result is wrong");
    }


    destroy_mat_full_fp32(&result_fp32);

    // exercise the old interface, too
    status = one_off_inmem(&table, &tree, "unweighted_fp64",
                                  false, 1.0, false, num_cores,
				  &result2);

    err(status != okay, "Compute failed");
    err(result2 == NULL, "Empty result");
    err(result2->n_samples != 6, "Wrong number of samples");

    for(unsigned int i = 0; i < (result2->n_samples*result2->n_samples); i++) {
        err(fabs(exp[i] - result2->matrix[i]) > 0.00001, "Result is wrong");
    }


    destroy_mat_full_fp64(&result2);

    status = one_off_inmem_fp32(&table, &tree, "unweighted_fp32",
                                    false, 1.0, false, num_cores,
				    &result2_fp32);

    err(status != okay, "Compute failed");
    err(result2 == NULL, "Empty result");
    err(result2_fp32->n_samples != 6, "Wrong number of samples");

    for(unsigned int i = 0; i < (result2_fp32->n_samples*result2_fp32->n_samples); i++) {
        err(fabs(exp[i] - result2_fp32->matrix[i]) > 0.00001, "Result is wrong");
    }


    destroy_mat_full_fp32(&result2_fp32);

}


int main(int argc, char** argv) {
    int num_cores = strtol(argv[1], NULL, 10);

    printf("Testing Striped UniFrac one_dense_pair...\n");
    test_su_dense(num_cores);
    printf("Testing Striped UniFrac one_off matrix...\n");
    test_su_matrix(num_cores);
    printf("Tests passed.\n");
    return 0;
}

