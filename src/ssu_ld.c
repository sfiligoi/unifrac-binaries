/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2023-2025, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

/* Handle pointing to the approriate libssu implementing the functionality
 * Initialized on first use. */
static void *dl_handle = NULL;

static void ssu_load(const char *fncname,
                     void **dl_ptr) {
   char *error;

   if (dl_handle==NULL) {
       dl_handle = dlopen(ssu_get_lib_name(), RTLD_LAZY);
       if (!dl_handle) {
          fputs(dlerror(), stderr);
          exit(1);
       }
   }

   *dl_ptr = dlsym(dl_handle, fncname);
   if ((error = dlerror()) != NULL)  {
       fputs(error, stderr);
       exit(1);
   }
}

static pthread_mutex_t dl_mutex = PTHREAD_MUTEX_INITIALIZER;

static void cond_ssu_load(const char *fncname,
                     void **dl_ptr) {

   pthread_mutex_lock(&dl_mutex);
   if ((*dl_ptr)==NULL) ssu_load(fncname,dl_ptr);
   pthread_mutex_unlock(&dl_mutex);
}


