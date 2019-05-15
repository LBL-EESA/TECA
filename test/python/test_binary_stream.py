#!/usr/bin/env python
import sys
import os
from teca import *
from mpi4py import *
from math import pi,e

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

if n_ranks < 2:
    sys.stderr.write('ERROR: test requires at least 2 ranks\n')
    sys.exit(-1)

baseline = sys.argv[1]

if len(sys.argv) != 2:
    sys.stderr.write('ERROR:\ntest_binary_stream.py [baseline]\n\n')
    sys.exit(-1)

master_rank = 0
worker_rank = n_ranks - 1
if rank == 0:
    sys.stderr.write('n_ranks = %d master_rank = %d ' \
        'worker_rank = %d\n'%(n_ranks, master_rank, worker_rank))


# generate a table
table = teca_table.New()

table.declare_columns(['A','B','C','D'],['f','d','i','l'])

table << .1 << .2 << 10 << 20 \
    << .2 << .4 << 20 << 40 \
    << pi << e << 30 << 60 \
    << 1.234e2 << 2.345e2 << 40 << 80 \
    << 1.234e-2 << 2.345e-2 << -50 << -100


bs = teca_binary_stream()

source = teca_dataset_source.New()

if rank == master_rank:
    # serialize the table
    table.to_stream(bs)

    # send to rank 1 for processing
    comm.send(bs.get_data(), dest=worker_rank, tag=23)

    # receive processed data back
    bs.clear()
    tmp = comm.recv(source=worker_rank, tag=27)
    bs.set_data(tmp)

    # deserialize into a new object
    table = teca_table.New()
    table.from_stream(bs)

    sys.stderr.write("=\n")
    sys.stderr.write("%s"%(str(table)))

    # feed the regression test with the updated table
    source.set_dataset(table)

if rank == worker_rank:
    # receive the seriealzed table
    tmp = comm.recv(source=master_rank, tag=23)
    bs.set_data(tmp)

    # generate the test table
    nums = teca_table.New()
    nums.declare_columns(['A','B','C','D'],['i','i','i','i'])

    # deserialize into a new object
    table = teca_table.New()
    table.from_stream(bs)

    sys.stderr.write("%s"%(str(table)))

    # modify the table in a predictable way
    nr = table.get_number_of_rows()
    nc = table.get_number_of_columns()
    j = 0
    while j < nr:
        i = 0
        while i < nc:
            q = j*nc + i
            nums[j,i] = q
            table[j,i] += nums[j,i]
            i += 1
        j += 1

    sys.stderr.write("+\n")
    sys.stderr.write("%s"%(str(nums)))

    # serialize the modified table
    bs.clear()
    table.to_stream(bs)
    tmp = bs.get_data()

    # send the modified table back
    comm.send(bs.get_data(), dest=master_rank, tag=27)


if os.path.exists(baseline):
    table_reader = teca_table_reader.New()
    table_reader.set_file_name(baseline)
    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, table_reader.get_output_port())
    diff.set_input_connection(1, source.get_output_port())
    diff.set_executive(teca_index_executive.New())
    diff.update()
else:
    sys.stderr.write('generating baseline\n')
    table_writer = teca_table_writer.New()
    table_writer.set_input_connection(source.get_output_port())
    table_writer.set_file_name(baseline)
    table_writer.set_executive(teca_index_executive.New())
    table_writer.update()
