from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name='QQQ',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='QQQ._CUDA',
            sources=[
                'csrc/pybind.cpp',
                'csrc/qqq_gemm.cu'
            ],
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)