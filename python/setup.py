from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

project_name = "gutzwiller"
omp = False

cpp_dir = os.path.join(os.path.expanduser("~"), "Dropbox", "Projects", "C++", project_name)
def cpp_path(cpp_file):
    return os.path.join(cpp_dir, cpp_file)

source_files = []
for file in os.listdir(cpp_dir):
    if file.endswith(".cpp") and file != "main.cpp":
        source_files.append(cpp_path(file))




#armadillo_include_dir = os.path.join(os.path.expanduser("~"), "Armadillo", "usr", "include")

#openblas_include_dir = os.path.join(os.path.expanduser("~"), "OpenBLAS", "include")
# = os.path.join(os.path.expanduser("~"), "OpenBLAS", "lib")

include_dirs = [cpp_dir]#, openblas_include_dir, armadillo_include_dir]
library_dirs = []#[openblas_library_dir]

extra_compile_args = ["-std=c++11", "-O3", "-march=native"] #, "-DARMA_DONT_USE_WRAPPER"]
if omp:
    extra_compile_args.append('-fopenmp')
extra_link_args = ['-fopenmp'] if omp else None

setup(ext_modules = cythonize(Extension(
		"Gutzwiller",
        language="c++",
        sources=["Gutzwiller.pyx"] + source_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        libraries=["m", "gsl", "armadillo"],# "openblas", "lapack"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
      ), compiler_directives={'language_level': 3}
))