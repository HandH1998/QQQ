from setuptools import setup, find_packages
from torch.utils import cpp_extension
import subprocess, os, shutil
import pathlib
HERE = pathlib.Path(__file__).absolute().parent

def install_fast_hadamard():
    # install fast hadamard transform
    hadamard_dir = os.path.join(HERE, 'third-party/fast-hadamard-transform')
    pip = shutil.which('pip')
    retcode = subprocess.call([pip, 'install', '-v', '-e', hadamard_dir])

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line and not line.startswith('#')]

install_fast_hadamard()
requirements = parse_requirements('requirements.txt')

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
    install_requires=requirements,
)