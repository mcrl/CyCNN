from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='CyConv2d', # import CyConv2d
      ext_modules=[
        CUDAExtension('CyConv2d_cuda', [
            'cycnn.cpp',
            'cycnn_cuda.cu',
          ])
      ],
      cmdclass={
        'build_ext': BuildExtension
      })
