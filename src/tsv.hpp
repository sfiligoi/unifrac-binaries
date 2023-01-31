/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2023-2023, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */


#ifndef _UNIFRAC_TSV_H
#define _UNIFRAC_TSV_H

#include <vector>
#include <map>
#include <string>

namespace su {
    // simple, row-by-row TSV parser
    class tsv {
        public:
            /* default constructor
             *
             * @param filename The path to the TSV file to read
             */
            tsv(const std::string &filename);

            /* default destructor
             *
             * Closes the file descriptor and destroys the objects.
             */
            ~tsv();

            /* Read next line
             *
             * Split the fields
             * Returns empty vector when hitting EOF
             */
            std::vector<const char *> get_next_line();

        private:
            int fd;
            int buf_filled;
            int buf_used;
            char buf[4096];
    };

    // Indexed tsv parser
    class indexed_tsv {
        public:
            /* main constructor
             *
             * @param filename The path to the TSV file to read
             * @param n_filter_els Number of filter elements
             * @param filter_els Only load rows having this filter index
             *
             * Note: All filter elements must be unique
             */
            indexed_tsv(const std::string &i_filename,
                        uint32_t _n_filter_els, const char * const* filter_els);

            /* constructor variant
             *
             * @param filename The path to the TSV file to read
             * @param filter_els Only load rows having this filter index
             *
             * Note: All filter elements must be unique
             */
            indexed_tsv(const std::string &_filename,
                        const std::vector<std::string> &filter_els);

            /* Read TSV file and group by column
             *
             * @param column The column to use
             * @param grouping Store the grouping here, array of size n_filter_els
             * @param n_groups Number of distinct elements in the grouping (out)
             *
             */
            void read_grouping(const std::string &column, uint32_t *grouping, uint32_t &n_groups) const;
        private:
            const std::string filename;
            const uint32_t n_filter_els;
            std::map<const std::string,uint32_t> filter_map;
    };
}

#endif /* _UNIFRAC_TSV_H */
