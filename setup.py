import os
import re
import sys
import platform
import subprocess
import multiprocessing

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

# when compiled outside of the git repo we must set the version
# manually. Also note that these must be unique per upload to PyPi
# so be sure to use an 'rcX' for testing
teca_version = "6.0.0"

# Pre-release test procedure:
# python3 setup.py build_ext
# python3 setup.py install
# python3 setup.py sdist
# python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# python3 -mpip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple teca==X.Y.ZrcW

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):

    user_options = build_ext.user_options + [
        ('without-mpi', None, 'Disables MPI parallel features'),
        ('without-netcdf-mpi', None, 'Disables NetCDF "parallel 4" features'),
        ('with-cray-mpich', None, 'Sets the build up to use Cray MPICH'),
        ('with-cuda', None, 'Sets the build up to use CUDA'),
        ('with-teca-data=', None, 'Path to the TECA_data subversion repo'),
        ('with-netcdf=', None, 'Sets the path to a NetCDF install. This may be '
                               'necessary on MacOS, Ubuntu, and other systems '
                               'with broken NetCDF packages'),
        ]

    boolean_options = build_ext.boolean_options + [
        'without-mpi', 'without-netcdf-mpi', 'with-cray-mpich'
        ]

    def initialize_options(self):
        super().initialize_options()

        self.without_mpi = False
        self.without_netcdf_mpi = False
        self.with_cray_mpich = False
        self.with_cuda = False
        self.with_teca_data = ''
        self.with_netcdf = ''

    def finalize_options(self):
        super().finalize_options()

        # enable the CMake logic that works with Cray MPICH and sets up CMake's
        # run time environment.
        if self.with_cray_mpich:
            self.without_mpi = False
            if 'CRAY_MPICH_DIR' in os.environ:
                cray_mpich_dir = os.environ['CRAY_MPICH_DIR']
                if 'PKG_CONFIG_PATH' in os.environ:
                    pkg_config_path = ':' + os.environ['PKG_CONFIG_PATH']
                else:
                    pkg_config_path = ''
                os.environ['PKG_CONFIG_PATH'] = cray_mpich_dir + \
                    '/lib/pkgconfig' + pkg_config_path
            else:
                sys.stderr.write('WARNING: Failed to add the CRAY_MPICH_DIR'
                                 ' to the PKG_CONFIG_PATH this is required on'
                                 ' NERSC Cori\n')

        # toggle NetCDF parallel 4 features based on MPI flags
        if self.without_mpi:
            self.without_netcdf_mpi = True

        elif self.with_netcdf:
            self.without_netcdf_mpi = False


    def run(self):

        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            raise RuntimeError('Windows is currrently unsupprted due to a lack of interest/funding')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        # this is where setuptools will look for our files to install
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # put verything in a directory below this for pip, if we don't do so
        # pip puts it all right in site-packages making a big fat mess
        extdir = os.path.join(extdir, 'teca')

        # set some flags for the cmake command line configuring this build
        # specifically we need to put the build where setuptools can find it
        # and also error out when dependencies are not found
        cmake_args = ['-DCMAKE_INSTALL_PREFIX=' + extdir,
                      '-DLIB_PREFIX=.', '-DREQUIRE_PYTHON=TRUE',
                      '-DTECA_PYTHON_VERSION=%d'%(sys.version_info.major),
                      '-DREQUIRE_UDUNITS=TRUE', '-DREQUIRE_NETCDF=TRUE',
                      '-DREQUIRE_BOOST=TRUE', '-DREQUIRE_OPENSSL=FALSE',
                      '-DREQUIRE_MPI=%s' % ('FALSE' if self.without_mpi else 'TRUE'),
                      '-DENABLE_CRAY_MPICH=%s' % ('TRUE' if self.with_cray_mpich else 'FALSE'),
                      '-DREQUIRE_NETCDF_MPI=%s' % ('FALSE' if self.without_netcdf_mpi else 'TRUE'),
                      '-DREQUIRE_CUDA=%s' % ('TRUE' if self.with_cuda else 'FALSE'),
                      '-DREQUIRE_TECA_DATA=%s' % ('TRUE' if self.with_teca_data else 'FALSE'),
                      '-DTECA_VERSION=%s(PyPi)'%(teca_version)]

        # override CMake search path for NetCDF. Currently necessary for MacOS
        # and Ubuntu due to broken or missing NetCDF parallel 4 capable
        # packages
        if self.with_netcdf:
            cmake_args += ['-DNETCDF_DIR=%s' % ( self.with_netcdf )]

        # point to the TECA_data repo
        if self.with_teca_data:
            cmake_args += ['-DTECA_DATA_ROOT=%s' % ( self.with_teca_data )]

        # set some command line arguments for cmake
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        install_args = ['--config', cfg]

        # figure out a reasonable number of cores for the build
        nj = multiprocessing.cpu_count()

        if platform.system() == "Windows":
            raise RuntimeError('Windows is currrently unsupprted due to a lack '
                               'of interest/funding')
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j%d'%(nj)]
            install_args += ['--', '-j%d'%(nj), 'install']

        # make the build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # builds teca using cmake and installs it where distutils is looking for it
        try:
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
            subprocess.check_call(['cmake', '--build', '.'] + install_args, cwd=self.build_temp)
        except:
            sys.stderr.write('\n')
            sys.stderr.write('================================================================\n')
            sys.stderr.write('  An error ocured building TECA. This is usually a symptom\n')
            sys.stderr.write('  of a missing dependency. The error messages above will indicate\n')
            sys.stderr.write('  which package is missing. Install the development package for\n')
            sys.stderr.write('  the missing dependency using your system\'s package manager\n')
            sys.stderr.write('================================================================\n')
            sys.stderr.write('\n')
            sys.tracebacklimit = 0
            raise

        # because we are putting everything in a subdirectory (see comment above re: pip)
        # write the init file that actually does the loading sets up the paths correctly
        f = open(os.path.join(extdir, '__init__.py'), 'w')
        f.write('import sys, os\n')
        f.write('sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n')
        f.write('from teca.teca_py import *\n')
        f.close()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='teca',
    version=teca_version,
    author='Burlen Loring',
    author_email='bloring@lbl.gov',
    description='The Toolkit for Extreme Climate Analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LBL-EESA/TECA",
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "License :: Other/Proprietary License",
        ],
    ext_modules=[CMakeExtension('teca')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=["mpi4py", "numpy", "matplotlib", "torch",],
    )

