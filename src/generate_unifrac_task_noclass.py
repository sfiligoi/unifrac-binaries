#
# Parse unifrac_task_impl.hpp
# and generate concrete implementations
# for all inline functions ending in _T
#

#
# Arguments:
#  $1 - namespace
#

import sys
nmspace = "su_%s"%sys.argv[1]

#
# ==========================
#

# replace template name with concrete type
def patch_type(s,t):
    return s.replace('TFloat',t).replace('TNum',t)

# extract argument name from the arg definition
def get_arg_name(s):
    return s.split()[-1].split('*')[-1]

#
# ==========================
#

def print_func_direct_noargs(ftype,nmspace,fname):
    print('%s %s::%s() {'%(ftype,nmspace,fname))
    if ftype=='void':
        print('  %s_T();'%fname);
    else:
        print('  return %s_T();'%fname)
    print('}');

def print_func_direct_args(ftype,nmspace,fname,ttype,fargs):
    print('template<>')
    print('%s %s::%s('%(ftype,nmspace,fname))
    for el in fargs[:-1]:
       print('\t\t\t%s,'%patch_type(el,ttype))
    print('\t\t\t%s) {'%patch_type(fargs[-1],ttype))

    print('  %s_T('%fname)
    for el in fargs[:-1]:
       print('\t%s,'%get_arg_name(el))
    print('\t%s);'%get_arg_name(fargs[-1]))

    print('}');

#
# ==========================
#

def print_func_indirect_noargs(ftype,nmspace,fname):
    print('%s %s::%s() {'%(ftype,nmspace,fname))
    if ftype=='void':
        print('  %s_%s();'%(nmspace,fname));
    else:
        print('  return %s_%s();'%(nmspace,fname))
    print('}');

def print_func_indirect_args(ftype,nmspace,fname,ttype,fargs):
    print('template<>')
    print('%s %s::%s('%(ftype,nmspace,fname))
    for el in fargs[:-1]:
       print('\t\t\t%s,'%patch_type(el,ttype))
    print('\t\t\t%s) {'%patch_type(fargs[-1],ttype))

    print('  %s_%s_%s('%(nmspace,fname,ttype))
    for el in fargs[:-1]:
       print('\t%s,'%get_arg_name(el))
    print('\t%s);'%get_arg_name(fargs[-1]))

    print('}');

#
# ==========================
#

def print_func_api_h_noargs(ftype,nmspace,fname):
    print('extern "C" %s %s_%s();'%(ftype,nmspace,fname))

def print_func_api_h_args(ftype,nmspace,fname,ttype,fargs):
    print('extern "C" %s %s_%s_%s('%(ftype,nmspace,fname,ttype))
    for el in fargs[:-1]:
       print('\t\t\t%s,'%patch_type(el,ttype))
    print('\t\t\t%s);'%patch_type(fargs[-1],ttype))

#
# ==========================
#

def print_func_api_noargs(ftype,nmspace,fname):
    print('%s %s_%s() {'%(ftype,nmspace,fname))
    if ftype=='void':
        print('  %s_T();'%fname);
    else:
        print('  return %s_T();'%fname)
    print('}');

def print_func_api_args(ftype,nmspace,fname,ttype,fargs):
    print('%s %s_%s_%s('%(ftype,nmspace,fname,ttype))
    for el in fargs[:-1]:
       print('\t\t\t%s,'%patch_type(el,ttype))
    print('\t\t\t%s) {'%patch_type(fargs[-1],ttype))

    print('  %s_T('%fname)
    for el in fargs[:-1]:
       print('\t%s,'%get_arg_name(el))
    print('\t%s);'%get_arg_name(fargs[-1]))

    print('}');

#
# ==========================
#

def print_func_noargs(ftype,nmspace,fname):
    method=sys.argv[2]
    if method=='direct':
        print_func_direct_noargs(ftype,nmspace,fname)
    elif method=='indirect':
        print_func_indirect_noargs(ftype,nmspace,fname)
    elif method=='api':
        print_func_api_noargs(ftype,nmspace,fname)
    elif method=='api_h':
        print_func_api_h_noargs(ftype,nmspace,fname)
    else:
        raise "Unknown generation method '%s'"%method

def print_func_args(ftype,nmspace,fname,ttype,fargs):
    method=sys.argv[2]
    if method=='direct':
        print_func_direct_args(ftype,nmspace,fname,ttype,fargs)
    elif method=='api':
        print_func_api_args(ftype,nmspace,fname,ttype,fargs)
    elif method=='indirect':
        print_func_indirect_args(ftype,nmspace,fname,ttype,fargs)
    elif method=='api_h':
        print_func_api_h_args(ftype,nmspace,fname,ttype,fargs)
    else:
        raise "Unknown generation method '%s'"%method

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

print('');

i=0
while (i<len(lines)):
    line = lines[i]
    i+=1
    if not line.startswith('static inline '):
        continue # not the beginning of an intersting function
    if line.find('_T(')<0:
        continue # not the beginning of an intersting function

    larr=line.split();
    ftype = larr[2];
    fname = larr[3].split('_T(')[0]

    print('// ==================================');
    if line.find('_T()')>=0:
        # special case, no arguments
        print_func_noargs(ftype,nmspace,fname)
        print('');
        continue

    # assuming ftype == void simplifies the code
    if ftype!='void':
        raise "Unsupported templated void found!"

    ftypes = set()
    fargs = []
    line = lines[i]
    i+=1
    while True:
        # get all the arguments up to the optional close
        larr=line.split(') ')[0].strip().split(',')
        for el in larr:
            el = el.strip() 
            if len(el)>0:
                fargs.append(el)
                if el.find('TFloat')>=0:
                    ftypes |= {'float','double'}
                elif el.find('TNum')>=0:
                    ftypes |= {'float','double','uint64_t','uint32_t','bool'}


        if line.find(') ')>=0:
            break # found end of args, exit the loop
        line = lines[i]
        i+=1
    
    for ft in ftypes:
        print_func_args(ftype,nmspace,fname,ft,fargs)
        print('');


