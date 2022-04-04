import os
import torch
# from torch.utils.ffi import create_extension
# from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import load

sources = ['src/slice_pool_layer.c']
headers = ['src/slice_pool_layer.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/slice_pool_layer_cuda.c']
    headers += ['src/slice_pool_layer_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

    this_file = os.path.dirname(os.path.realpath(__file__))
    print(this_file)
    extra_objects = ['src/cuda/slice_pool_layer_cuda_kernel.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

    # ffi = BuildExtension(
    #                        name='_ext.slice_pool_layer',
    #                        sources=sources,
    #                        with_cuda=with_cuda,
    #                        extra_include_paths=extra_objects
    #                        )
    ffi = load(
        name='_ext.slice_pool_layer',
        sources=sources,
        with_cuda=with_cuda,
        extra_include_paths=extra_objects)
# headers=headers,

else:
    print('NO CUDA')
    # ffi = BuildExtension(
    #                        '_ext.slice_pool_layer',
    #                        headers=headers,
    #                        sources=sources,
    #                        define_macros=defines,
    #                        relative_to=__file__,
    #                        with_cuda=with_cuda,
    #                        #extra_objects=extra_objects
    #                        )

if __name__ == '__main__':
    ffi.build()
