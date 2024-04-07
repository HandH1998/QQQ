from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name='QQQ',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='QQQ._CUDA',
            sources=[
                'csrc/int8gemm/cuda_utils.cc',
                'csrc/int8gemm/cublasINT8MMWrapper.cc',
                'csrc/int8gemm/cublasAlgoMap.cc',
                'csrc/int8gemm/bindings.cpp',
            ],
            extra_link_args=['-lcublas_static', '-lcublasLt_static',
                             '-lculibos', '-lcudart', '-lcudart_static',
                             '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/'],
            extra_compile_args={'cxx': ['-std=c++17', '-O3'],
                                'nvcc': ['-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']},
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)