from setuptools import setup, Extension

import numpy


setup(
    name="py_lsd",
    version="1.6",
    description="Python Package with lsd C extension",
    author="Xueyuan Chen",
    author_email="1829401081@stu.suda.edu.cn",
    url="https://www.github.com/Evlpsrfc/py_lsd",
    include_dirs=[numpy.get_include()],
    ext_modules=[Extension("lsd_ext", ["lsd_ext.c", "lsd_1.6/lsd.c"])]
)