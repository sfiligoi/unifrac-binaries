/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2023-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */


#include "tsv.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include <algorithm>
#include <stdexcept>

su::tsv::tsv(const std::string &filename)
  : fd(open(filename.c_str(),O_RDONLY))
  , buf_filled(0), buf_used(0) {
  if (fd>=0) {
    buf_filled=read(fd,buf,4096);
  }
}

su::tsv::~tsv() {
  if (fd>=0) {
    close(fd);
  }
}

std::vector<const char *> su::tsv::get_next_line() {
  if ((fd<0) || (buf_filled<=0)) {
    // EOF or error, return empty vector
    std::vector<const char *> empty;
    return empty;;
  }

  char * pnewline;
  if (buf_used<buf_filled) {
     // see if we have the whole line in buffer
     pnewline = std::find(buf+buf_used,buf+buf_filled,'\n');
     if (pnewline==(buf+buf_filled)) { //not found
       uint32_t diff = buf_filled-buf_used;
       // make space for more
       memmove(buf,buf+buf_used,diff);
       buf_used = 0;
       buf_filled = diff;
       // read more
       int cnt = read(fd,buf,4096-diff);
       if (cnt>0) buf_filled += cnt;
       // try again
       pnewline = std::find(buf,buf+buf_filled,'\n');
     }
  } else { // nothing left in buffer
    // read another chunk
    buf_used = 0;
    buf_filled=read(fd,buf,4096);
    // search
    pnewline = std::find(buf,buf+buf_filled,'\n');
  } 

  if (pnewline==(buf+buf_filled)) { // the line is longer than the buffer, not supported
    // abort
    throw std::overflow_error("TSV line too long");
  }

  // construct the outval
  int n_tabs = std::count(buf+buf_used,pnewline,'\t');
  std::vector<const char *> outval(n_tabs);

  // return pointers to buffer
  char *plast = buf+buf_used;
  outval[0] = plast;
  for (int i=1; i<n_tabs; i++) {
     char *pnext = std::find(plast,pnewline,'\t');
     pnext[0] = 0; // terminate prev string
     outval[i] = pnext;
     plast = pnext+1;
  }
  pnewline[0] = 0; // terminate last string

  return outval;
}

