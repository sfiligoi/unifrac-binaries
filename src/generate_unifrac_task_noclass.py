#
# Parse unifrac_task_impl.hpp
# and generate concrete implementations
# for all inline functions ending in _T
#

# replace template name with concrete type
def patch_type(s,t):
    return s.replace('TFloat',t).replace('TNum',t)

# extract argument name from the arg definition
def get_arg_name(s):
    return s.split()[-1].split('*')[-1]

# print out the header
print('// Generated from unifrac_task_impl.hpp');
print('// Do not edit by hand');
print('');
print('#include "unifrac_task_noclass.hpp"');
print('#include "unifrac_task_impl.hpp"');
print('');


with open('unifrac_task_impl.hpp','r') as fd:
    lines=fd.readlines()

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
        print('%s SUCMP_NM::%s() {'%(ftype,fname))
        if ftype=='void':
            print('  %s_T();'%fname);
        else:
            print('  return %s_T();'%fname)
        print('}');
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
        print('template<>')
        print('%s SUCMP_NM::%s('%(ftype,fname))
        for el in fargs[:-1]:
           print('\t\t\t%s,'%patch_type(el,ft))
        print('\t\t\t%s) {'%patch_type(fargs[-1],ft))

        print('  %s_T('%fname)
        for el in fargs[:-1]:
           print('\t%s,'%get_arg_name(el))
        print('\t%s);'%get_arg_name(fargs[-1]))

        print('}');
    print('');


