/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2016-2025, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

/* Note: Allow multiple definitions of this header, using different SUCMP_NM */

#include "task_parameters.hpp"
#include "tree.hpp"
#include "biom_interface.hpp"

#include "unifrac_internal.hpp"

#ifndef SUCMP_NM
/* create a default */
#define SUCMP_NM su_cpu
#endif

#pragma omp requires unified_address
#pragma omp requires unified_shared_memory
namespace SUCMP_NM {

  // Returns True iff a GPU can be used
  bool found_gpu();

  void unifrac(const su::biom_interface &table,
               const su::BPTree &tree,
               su::Method unifrac_method,
               std::vector<double*> &dm_stripes,
               std::vector<double*> &dm_stripes_total,
               const su::task_parameters* task_p);

  void unifrac_vaw(const su::biom_interface &table,
                   const su::BPTree &tree,
                   su::Method unifrac_method,
                   std::vector<double*> &dm_stripes,
                   std::vector<double*> &dm_stripes_total,
                   const su::task_parameters* task_p);

}

