import sys
from teca import *

sys.stderr.write('TECA_VERSION_DESCR=%s\n'%(get_teca_version_descr()))
sys.stderr.write('TECA_PYTHON_VERSION=%d\n'%(get_teca_python_version()))
sys.stderr.write('TECA_HAS_REGEX=%s\n'%(str(get_teca_has_regex())))
sys.stderr.write('TECA_HAS_NETCDF=%s\n'%(str(get_teca_has_netcdf())))
sys.stderr.write('TECA_HAS_MPI=%s\n'%(str(get_teca_has_mpi())))
sys.stderr.write('TECA_HAS_BOOST=%s\n'%(str(get_teca_has_boost())))
sys.stderr.write('TECA_HAS_VTK=%s\n'%(str(get_teca_has_vtk())))
sys.stderr.write('TECA_HAS_PARAVIEW=%s\n'%(str(get_teca_has_paraview())))
sys.stderr.write('TECA_HAS_UDUNITS=%s\n'%(str(get_teca_has_udunits())))
sys.stderr.write('TECA_HAS_OPENSSL=%s\n'%(str(get_teca_has_openssl())))
sys.stderr.write('TECA_HAS_DATA=%s\n'%(str(get_teca_has_data())))
sys.stderr.write('TECA_DATA_ROOT=%s\n'%(get_teca_data_root()))
