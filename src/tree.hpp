/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2016-2021, UniFrac development team.
 * All rights reserved.
 *
 * See LICENSE file for more details
 */

#ifndef __UNIFRAC_TREE_H
#define __UNIFRAC_TREE_H 1

#include <cstdint>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>

namespace su {
    class BPTree {
        public:
            /* nullary constructor */
            BPTree();

            /* tracked attributes */
            std::vector<double> lengths;
            std::vector<std::string> names;

            /* total number of parentheses */
            uint32_t nparens;

            /* default constructor
             *
             * @param newick A newick string
             */
            BPTree(const std::string& newick);
            BPTree(const char * newick);
            
            /* constructor from a defined topology 
             *
             * @param input_structure A boolean vector defining the topology
             * @param input_lengths A vector of double of the branch lengths
             * @param input_names A vector of str of the vertex names
             */
            BPTree(const std::vector<bool>& input_structure, const std::vector<double>& input_lengths, const std::vector<std::string>& input_names);
            ~BPTree();

            /* constructor from a defined topology using c-types
             *
             * @param input_structure A boolean array defining the topology
             * @param input_lengths A double array of the branch lengths
             * @param input_names A char* array of the names
             * @param n_parens The length of the topology
             */
            BPTree(const bool* input_structure, const double* input_lengths, const char* const * input_names, const int n_parens);

            /* postorder tree traversal
             *
             * Get the index position of the ith node in a postorder tree
             * traversal.
             *
             * @param i The ith node in a postorder traversal
             */
            uint32_t postorderselect(uint32_t i)const ;

            /* preorder tree traversal
             *
             * Get the index position of the ith node in a preorder tree
             * traversal.
             *
             * @param i The ith node in a preorder traversal
             */
            uint32_t preorderselect(uint32_t i) const;

            /* Test if the node at an index position is a leaf
             *
             * @param i The node to evaluate
             */
            bool isleaf(uint32_t i) const;

            /* Get the left child of a node
             *
             * @param i The node to obtain the left child from
             */
            uint32_t leftchild(uint32_t i) const ;

            /* Get the right child of a node
             *
             * @param i The node to obtain the right child from
             */
            uint32_t rightchild(uint32_t i) const;

            /* Get the right sibling of a node
             *
             * @param i The node to obtain the right sibling from
             */
            uint32_t rightsibling(uint32_t i) const;
            
            /* Get the parent of a node
             *
             * @param i The node to obtain the parent of
             */
            int32_t parent(uint32_t i) const;

            /* get the names at the tips of the tree */
            std::unordered_set<std::string> get_tip_names() const;

            /* public getters */
            const std::vector<bool>& get_structure() const;
            const std::vector<uint32_t>& get_openclose() const;

            /* serialize the structure as a sequence of 1s and 0s */
            void print() {
                for(auto c = structure.begin(); c != structure.end(); c++) {
                    if(*c)
                        std::cout << "1";
                    else
                        std::cout << "0";
                }
                std::cout << std::endl;
            }
            BPTree mask(const std::vector<bool>& topology_mask, const std::vector<double>& in_lengths) const; // mask self

            BPTree shear(std::unordered_set<std::string> to_keep) const;

            BPTree collapse() const;

        private:
            std::vector<bool> structure;          // the topology
            std::vector<uint32_t> openclose;      // cache'd mapping between parentheses
            std::vector<uint32_t> select_0_index; // cache of select 0
            std::vector<uint32_t> select_1_index; // cache of select 1
            std::vector<uint32_t> excess;

            void index_and_cache();  // construct the select caches
            void newick_to_bp(const char* newick);  // convert a newick string to parentheses
            void newick_to_bp(const std::string& newick);  // convert a newick string to parentheses
            void newick_to_metadata(const char* newick);  // convert newick to attributes
            void newick_to_metadata(const std::string& newick);  // convert newick to attributes
            void structure_to_openclose();  // set the cache mapping between parentheses pairs
            void set_node_metadata(unsigned int open_idx, std::string &token); // set attributes for a node
            static bool is_structure_character(char c);  // test if a character is a newick structure
            inline uint32_t open(uint32_t i) const;  // obtain the index of the opening for a given parenthesis
            inline uint32_t close(uint32_t i) const;  // obtain the index of the closing for a given parenthesis
            static std::string tokenize(const char * &start, const char * const end);  // newick -> tokens

            int32_t bwd(uint32_t i, int32_t d) const;
            int32_t enclose(uint32_t i) const;

	    // to be used in the constructors only
            template<typename T> void _init(T newick);
    };
}

#endif /* UNIFRAC_TREE_H */

