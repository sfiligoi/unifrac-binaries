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
#include <string>

namespace su {
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
}

#endif /* _UNIFRAC_TSV_H */
