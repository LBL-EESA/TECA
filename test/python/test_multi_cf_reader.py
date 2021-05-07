try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except:
    rank = 0
    n_ranks = 1
from teca import *
import sys

set_stack_trace_on_error()
set_stack_trace_on_mpi_error()

def parse_args(args):
    """
    Parse the command tail. See -h for an explanation.
    """
    first_step = 0
    last_step = -1
    regex = []
    var = []
    out_file = ''
    baseline = ''
    time_reader = ''
    geometry_reader = ''
    config_file = ''
    verbose = 0
    tmp_vars = []
    in_list = False
    it = iter(args)
    while True:
        try:
            arg = next(it)

            if arg == ')':
                in_list = False
                var.append(tmp_vars)

            elif in_list:
                tmp_vars.append(arg)

            elif arg == '(':
                in_list = True
                tmp_vars = []
                try:
                    regex.append(next(it))
                except Exception:
                    raise RuntimeError('Missing regex at begining of list')

            elif arg == '-v':
                verbose = 1

            elif arg == '-t':
                try:
                    time_reader = next(it)
                except Exception:
                     raise RuntimeError('Missing time reader key')

            elif arg == '-g':
                try:
                    geometry_reader = next(it)
                except Exception:
                     raise RuntimeError('Missing geometry reader key')

            elif arg == '-o':
                try:
                    out_file = next(it)
                except Exception:
                    raise RuntimeError('Missing out file name')

            elif arg == '-b':
                try:
                    baseline = next(it)
                except Exception:
                    raise RuntimeError('Missing base file name')

            elif arg == '-s':
                try:
                    first_step = next(it)
                except Exception:
                    raise RuntimeError('Missing base file name')

            elif arg == '-e':
                try:
                    last_step = next(it)
                except Exception:
                    raise RuntimeError('Missing base file name')

            elif arg == '-f':
                try:
                    config_file = next(it)
                except Exception:
                    raise RuntimeError('Missing config_file name')

            elif arg == '-h':
                sys.stderr.write('usage: test_multi_cf_reader [-o out file] '
                                 '[-b base line] [-f config file ] [-s first step] '
                                 ' [-e last step] [-v] [( regex var0 ... varn )] '
                                 ' ... [( regex var0 ... varn )]\n')
                sys.exit(-1)

        except StopIteration:
            return regex, time_reader, geometry_reader, \
                var, config_file, first_step, last_step, out_file, \
                baseline, verbose


# parse the comman tail
regex, time_reader, geometry_reader, \
var, config_file, first_step, last_step, \
out_file, baseline, verbose = parse_args(sys.argv)

if not config_file:
    if not geometry_reader:
        sys.stderr.write('ERROR: no geometry_reader was specified\n')
        sys.exit(-1)

    if not time_reader:
        sys.stderr.write('ERROR: no time_reader was specified\n')
        sys.exit(-1)

    if not len(regex):
        sys.stderr.write('ERROR: at least one regex must be provided\n')
        sys.exit(-1)

# read data from multiple files, present it as a single dataset
cfmr = teca_multi_cf_reader.New()
cfmr.set_x_axis_variable('lon')
cfmr.set_y_axis_variable('lat')
cfmr.set_z_axis_variable('plev')
cfmr.set_t_axis_variable('time')
if config_file:
    cfmr.set_input_file(config_file)
    all_var = cfmr.get_variables()
else:
    n = len(regex)
    i = 0
    while i < n:
        key = 'r_%d'%(i)
        cfmr.add_reader(regex[i], key, 0, 0, var[i])
        i += 1
    cfmr.set_time_reader(time_reader)
    cfmr.set_geometry_reader(geometry_reader)

    all_var = []
    for vl in var:
        for v in vl:
            all_var.append(v)

if verbose:
    if config_file:
        sys.stderr.write('config_file=%s\n'%(config_file))
        sys.stderr.write('var=%s\n'%(str(all_var)))
    else:
        sys.stderr.write('regex=%s\n'%(str(regex)))
        sys.stderr.write('time_reader=%s\n'%(str(time_reader)))
        sys.stderr.write('geometry_reader=%s\n'%(str(geometry_reader)))
        sys.stderr.write('var=%s\n'%(str(var)))
    sys.stderr.write('first_step=%s\n'%(str(first_step)))
    sys.stderr.write('last_step=%s\n'%(str(last_step)))
    sys.stderr.write('out_file=%s\n'%(str(out_file)))
    sys.stderr.write('baseline=%s\n'%(str(baseline)))

    md = cfmr.update_metadata()
    sys.stderr.write('md = %s\n'%(str(md)))

coords = teca_normalize_coordinates.New()
coords.set_input_connection(cfmr.get_output_port())

exe = teca_index_executive.New()
exe.set_start_index(first_step)
exe.set_end_index(last_step)
exe.set_arrays(all_var)
exe.set_verbose(verbose)

# write the dataset as a sintle file
wri = teca_cf_writer.New()
wri.set_input_connection(coords.get_output_port())
wri.set_verbose(verbose)
wri.set_executive(exe)
wri.set_thread_pool_size(1)
wri.set_point_arrays(all_var)
wri.set_steps_per_file(256)
wri.set_file_name(out_file)

wri.update()
