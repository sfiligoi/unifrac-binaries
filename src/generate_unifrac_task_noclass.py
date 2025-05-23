#
# Parse unifrac_task_impl.hpp
# and generate concrete implementations
# for all inline functions ending in _T
#

#
# Arguments:
#  $1 - namespace
#  $2 - method
#

from ssu_generate_helper import print_body
import sys
nmspace = "su_%s"%sys.argv[1]
method=sys.argv[2]

#
# ==========================
#

with open('unifrac_task_impl.hpp','r') as fd:
    lines=fd.readlines()

#
# ==========================
#

# print out the header
print('// Generated from unifrac_task_impl.hpp (using method %s)'%sys.argv[2]);
print('// Do not edit by hand');
print('');

if sys.argv[2] in ('direct','indirect',):
    # we are generating unifrac_task_noclass.cpp
    print('#include "unifrac_task_noclass.hpp"');

if sys.argv[2] in ('indirect','api',):
    # we referencing the api
    print('#include "unifrac_task_api_%s.h"'%sys.argv[1]);

if sys.argv[2] in ('direct','api',):
    # we are generating the actual code
    print('#include "unifrac_task_impl.hpp"');

if sys.argv[2] in ('api_h',):
    # bool and unit_t are not standard in C without these header
    print('#include <stdbool.h>')
    print('#include <stdint.h>')

if sys.argv[2] in ('indirect',):
    # function expected by ssu_ld
    print('static const char *ssu_get_lib_name() { return "libssu_%s.so";}'%sys.argv[1])
    print('#include "ssu_ld.c"')

print('');

print_body(method, lines, nmspace)

