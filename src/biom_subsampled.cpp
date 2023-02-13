/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2023-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include "biom_subsampled.hpp"

using namespace su;

biom_subsampled::biom_subsampled(const biom_inmem &parent, const uint32_t n) 
  : biom_inmem(parent, true)
{}

biom_subsampled::~biom_subsampled()
{}
