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

void test_su(int num_cores){
    mat_t* result = NULL;
    const char* table = "test.biom";
    const char* tree = "test.tre";
    const char* method = "unweighted";
    const double exp[] = {0.2, 0.57142857, 0.6, 0.5, 0.2, 0.42857143, 0.66666667, 0.6, 0.33333333, 0.71428571, 0.85714286, 0.42857143, 0.33333333, 0.4, 0.6};
    
    ComputeStatus status;
    status = one_off(table, tree, method,
                     false, 1.0, false, num_cores, &result);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result->n_samples != 6, "Wrong number of samples");
    err(result->cf_size != 15, "Wrong condensed form size");
    err(!result->is_upper_triangle, "Result is not squaure");

    for(unsigned int i = 0; i < result->cf_size; i++) 
        err(fabs(exp[i] - result->condensed_form[i]) > 0.00001, "Result is wrong");

}

void test_su_wtree(int num_cores){
    mat_t* result = NULL;
    const char* table = "test.biom";
    const char* tree = "test.tre";
    const char* method = "unweighted";
    const double exp[] = {0.2, 0.57142857, 0.6, 0.5, 0.2, 0.42857143, 0.66666667, 0.6, 0.33333333, 0.71428571, 0.85714286, 0.42857143, 0.33333333, 0.4, 0.6};
    
    opaque_bptree_t *tree_data;
    IOStatus iostatus;
    iostatus = read_bptree_opaque(tree, &tree_data);
    err(iostatus != read_okay, "Tree read failed");
    int tree_obs = get_bptree_opaque_els(tree_data);
    err(tree_obs != 5, "Wrong number of obs");

    ComputeStatus status;
    status = one_off_wtree(table, tree_data, method,
                     false, 1.0, false, num_cores, &result);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result->n_samples != 6, "Wrong number of samples");
    err(result->cf_size != 15, "Wrong condensed form size");
    err(!result->is_upper_triangle, "Result is not squaure");

    for(unsigned int i = 0; i < result->cf_size; i++) 
        err(fabs(exp[i] - result->condensed_form[i]) > 0.00001, "Result is wrong");

    destroy_bptree_opaque(&tree_data);
}

void test_su_wtree2(int num_cores){
    mat_t* result = NULL;
    const char* table = "test.biom";
    const char tree_str[] = "(GG_OTU_1:1,(GG_OTU_2:1,GG_OTU_3:1):1,(GG_OTU_5:1,GG_OTU_4:1):1);";
    const char* method = "unweighted";
    const double exp[] = {0.2, 0.57142857, 0.6, 0.5, 0.2, 0.42857143, 0.66666667, 0.6, 0.33333333, 0.71428571, 0.85714286, 0.42857143, 0.33333333, 0.4, 0.6};
    
    opaque_bptree_t *tree_data;
    load_bptree_opaque(tree_str, &tree_data);
    int tree_obs = get_bptree_opaque_els(tree_data);
    err(tree_obs != 5, "Wrong number of obs");

    ComputeStatus status;
    status = one_off_wtree(table, tree_data, method,
                     false, 1.0, false, num_cores, &result);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result->n_samples != 6, "Wrong number of samples");
    err(result->cf_size != 15, "Wrong condensed form size");
    err(!result->is_upper_triangle, "Result is not squaure");

    for(unsigned int i = 0; i < result->cf_size; i++) 
        err(fabs(exp[i] - result->condensed_form[i]) > 0.00001, "Result is wrong");

    destroy_bptree_opaque(&tree_data);
}

void test_su_matrix(int num_cores){
    mat_full_fp32_t* result = NULL;
    const char* table = "test.biom";
    const char* tree = "test.tre";
    const char* method = "unweighted_fp32";
    float exp[] = { 0.0,        0.2,        0.57142857, 0.6,        0.5,        0.2, 
                    0.2,        0.0,        0.42857143, 0.66666667, 0.6,        0.33333333,
		    0.57142857, 0.42857143, 0.0,        0.71428571, 0.85714286, 0.42857143, 
		    0.6,        0.66666667, 0.71428571, 0.0,        0.33333333, 0.4,
		    0.5,        0.6,        0.85714286, 0.33333333, 0.0,        0.6,
                    0.2,        0.33333333, 0.42857143, 0.4,        0.6,        0.0        };
    
    ComputeStatus status;
    status = one_off_matrix_fp32_v2(table, tree, method,
                                    false, 1.0, false, num_cores,
                                    0, true, NULL,
				    &result);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result->n_samples != 6, "Wrong number of samples");

    for(unsigned int i = 0; i < (result->n_samples*result->n_samples); i++) {
        err(fabs(exp[i] - result->matrix[i]) > 0.00001, "Result is wrong");
    }


    destroy_mat_full_fp32(&result);
}

void test_su_matrix_wtree(int num_cores){
    mat_full_fp32_t* result = NULL;
    const char* table = "test.biom";
    const char* tree = "test.tre";
    const char* method = "unweighted";
    float exp[] = { 0.0,        0.2,        0.57142857, 0.6,        0.5,        0.2, 
                    0.2,        0.0,        0.42857143, 0.66666667, 0.6,        0.33333333,
		    0.57142857, 0.42857143, 0.0,        0.71428571, 0.85714286, 0.42857143, 
		    0.6,        0.66666667, 0.71428571, 0.0,        0.33333333, 0.4,
		    0.5,        0.6,        0.85714286, 0.33333333, 0.0,        0.6,
                    0.2,        0.33333333, 0.42857143, 0.4,        0.6,        0.0        };
    
    opaque_bptree_t *tree_data;
    IOStatus iostatus;
    iostatus = read_bptree_opaque(tree, &tree_data);
    err(iostatus != read_okay, "Tree read failed");

    ComputeStatus status;
    status = one_off_matrix_fp32_v2t(table, tree_data, method,
                                    false, 1.0, false, num_cores,
                                    0, true, NULL,
				    &result);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result->n_samples != 6, "Wrong number of samples");

    for(unsigned int i = 0; i < (result->n_samples*result->n_samples); i++) 
        err(fabs(exp[i] - result->matrix[i]) > 0.00001, "Result is wrong");


    destroy_mat_full_fp32(&result);
    destroy_bptree_opaque(&tree_data);
}

void test_su_matrix_wtree2(int num_cores){
    mat_full_fp32_t* result = NULL;
    const char* table = "test.biom";
    const char tree_str[] = "(GG_OTU_1:1,(GG_OTU_2:1,GG_OTU_3:1):1,(GG_OTU_5:1,GG_OTU_4:1):1); \t "; //add final space, to exercise edge case
    const char* method = "unweighted";
    float exp[] = { 0.0,        0.2,        0.57142857, 0.6,        0.5,        0.2, 
                    0.2,        0.0,        0.42857143, 0.66666667, 0.6,        0.33333333,
		    0.57142857, 0.42857143, 0.0,        0.71428571, 0.85714286, 0.42857143, 
		    0.6,        0.66666667, 0.71428571, 0.0,        0.33333333, 0.4,
		    0.5,        0.6,        0.85714286, 0.33333333, 0.0,        0.6,
                    0.2,        0.33333333, 0.42857143, 0.4,        0.6,        0.0        };
    
    opaque_bptree_t *tree_data;
    load_bptree_opaque(tree_str, &tree_data);

    ComputeStatus status;
    status = one_off_matrix_fp32_v2t(table, tree_data, method,
                                    false, 1.0, false, num_cores,
                                    0, true, NULL,
				    &result);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result->n_samples != 6, "Wrong number of samples");

    for(unsigned int i = 0; i < (result->n_samples*result->n_samples); i++) 
        err(fabs(exp[i] - result->matrix[i]) > 0.00001, "Result is wrong");


    destroy_mat_full_fp32(&result);
    destroy_bptree_opaque(&tree_data);
}

void test_su_dense(int num_cores){
    double result = 0.0;
    const char tree_str[] = "(GG_OTU_1:1,(GG_OTU_2:1,GG_OTU_3:1):1,(GG_OTU_5:1,GG_OTU_4:1):1);";
    const char* table_oids[] = { "GG_OTU_1", "GG_OTU_2", "GG_OTU_3", "GG_OTU_4", "GG_OTU_5" };
    const double sample1[] = { 0, 5, 0, 2, 0 };
    const double sample2[] = { 0, 1, 0, 1, 1 };
    const double sample3[] = { 1, 0, 1, 1, 1 };
    const char* method = "unweighted";
    float exp13 = 0.57142857;
    float exp21 = 0.2;
    
    opaque_bptree_t *tree_data;
    load_bptree_opaque(tree_str, &tree_data);

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
}

void test_faith_pd(){
    r_vec* result = NULL;
    const char* table = "test.biom";
    const char* tree = "test.tre";
    const double exp[] = {4, 5, 6, 3, 2, 5};

    ComputeStatus status;
    status = faith_pd_one_off(table, tree, &result);

    err(status != okay, "Compute failed");
    err(result == NULL, "Empty result");
    err(result->n_samples != 6, "Wrong number of samples");

    for(unsigned int i = 0; i < result->n_samples; i++)
        err(fabs(exp[i] - result->values[i]) > 0.00001, "Result is wrong");

}

int main(int argc, char** argv) {
    int num_cores = strtol(argv[1], NULL, 10);

    printf("Testing Striped UniFrac one_off...\n");
    test_su(num_cores);
    printf("Testing Striped UniFrac one_off_wtree...\n");
    test_su_wtree(num_cores);
    test_su_wtree2(num_cores);
    printf("Testing Striped UniFrac one_off matrix...\n");
    test_su_matrix(num_cores);
    printf("Testing Striped UniFrac one_off matrix_wtree...\n");
    test_su_matrix_wtree(num_cores);
    test_su_matrix_wtree2(num_cores);
    printf("Testing Striped UniFrac one_dense_pair...\n");
    test_su_dense(num_cores);
    printf("Tests passed.\n");
    printf("Testing Faith's PD...\n");
    test_faith_pd();
    printf("Tests passed.\n");
    return 0;
}

