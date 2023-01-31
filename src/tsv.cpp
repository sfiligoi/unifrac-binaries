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
    return empty;
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
       int cnt = read(fd,buf+buf_filled,4096-buf_filled);
       if (cnt>0) buf_filled += cnt;
       // try again
       pnewline = std::find(buf,buf+buf_filled,'\n');
     }
  } else { // nothing left in buffer
    // read another chunk
    buf_used = 0;
    buf_filled=read(fd,buf,4096);
    if (buf_filled<=0) {
      // EOF or error, return empty vector
      std::vector<const char *> empty;
      return empty;
    }
    // search
    pnewline = std::find(buf,buf+buf_filled,'\n');
  } 

  if (pnewline==(buf+buf_filled)) { // the line is longer than the buffer, not supported
    // abort
    throw std::overflow_error("TSV line too long");
  }

  // construct the outval
  int n_tabs = std::count(buf+buf_used,pnewline,'\t');
  std::vector<const char *> outval(n_tabs+1);

  // return pointers to buffer
  char *plast = buf+buf_used;
  outval[0] = plast;
  for (int i=0; i<n_tabs; i++) {
     char *pnext = std::find(plast,pnewline,'\t');
     pnext[0] = 0; // terminate prev string
     outval[i+1] = pnext+1;
     plast = pnext+1;
  }
  pnewline[0] = 0; // terminate last string
  buf_used = pnewline+1-buf;

  return outval;
}

// ===================== indexed_tsv ======================


su::indexed_tsv::indexed_tsv(const std::string &_filename,
                             uint32_t _n_filter_els, const char * const* filter_els)
   : filename(_filename)
   , n_filter_els(_n_filter_els)
   , filter_map() {
  for (uint32_t i=0; i<n_filter_els; i++) {
     filter_map[filter_els[i]]=i;
  }
}

su::indexed_tsv::indexed_tsv(const std::string &_filename,
                             const std::vector<std::string>& filter_els)
   : filename(_filename)
   , n_filter_els(filter_els.size())
   , filter_map() {
  for (uint32_t i=0; i<n_filter_els; i++) {
     filter_map[filter_els[i]]=i;
  }
}


void su::indexed_tsv::read_grouping(const std::string &column, uint32_t *grouping, uint32_t &n_groups) const {
  su::tsv tsv_obj(filename);

  // first find the column id
  uint32_t column_idx = 0;
  {
     // first line is the header
     std::vector<const char *> header_line = tsv_obj.get_next_line();
     for (int i=1; i< header_line.size(); i++) {
       if (column==header_line[i]) {
          column_idx = i;
          break; // found
       }
     }
     if (column_idx<1) { // 1st column is the index
       throw std::runtime_error("Column not found");
     }
  }

  // there can be at most n_filter_els indexes, so this makes the values invalid
  for (int i=1; i< n_filter_els; i++) grouping[i] = n_filter_els;

  // Gouping needs a number, not the value
  // Will build it as we read
  std::map<const std::string, uint32_t> column_map;
  uint32_t last_grouping_nr = 0;

  while (true) {
     std::vector<const char *> row = tsv_obj.get_next_line();
     if (row.empty()) break; // reached EOF

     const std::string idx_val = row[0];
     auto row_itr = filter_map.find(idx_val);
     if (row_itr == filter_map.end()) {
       // not in the whitelist, ignore
       continue;
     }
     uint32_t row_idx = row_itr->second;

     const std::string column_val = row.at(column_idx); // will throw, if not found/valid
     uint32_t column_grouping_nr;
     auto column_itr = column_map.find(column_val);
     if (column_itr != column_map.end()) {
       // found
       column_grouping_nr = column_itr->second;
     } else {
       // new element
       column_grouping_nr     = last_grouping_nr;
       column_map[column_val] = last_grouping_nr;
       last_grouping_nr++;
     }
     grouping[row_idx] = column_grouping_nr;
  }

  // make sure we actually got all the filter_els
  for (int i=1; i< n_filter_els; i++) {
     if (grouping[i] == n_filter_els) throw std::runtime_error("Not all elements found");
  }

  n_groups = last_grouping_nr;
}


