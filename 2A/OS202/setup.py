"""
this code is setup to run MPI code
mpirun -n 8 /bin/python3 /home/tr0fin0/Downloads/tmp_OS202/setup.py
"""

from mpi4py import MPI

print(f'rank: {MPI.COMM_WORLD.Get_rank()} processes: {MPI.COMM_WORLD.Get_size()}')


# MPI.data_type
# https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.html
MPI.BYTE
MPI.CHAR
MPI.SHORT
MPI.INT
MPI.LONG
MPI.FLOAT
MPI.DOUBLE
MPI.BOOL

# MPI.OPERATION
MPI.MAX
MPI.MIN
MPI.MAXLOC
MPI.MINLOC
MPI.SUM
MPI.PROD

# commands
# https://mpi4py.readthedocs.io/en/stable/tutorial.html
# no-buffer
MPI.COMM_WORLD.bcast(data, root=process)
MPI.COMM_WORLD.scatter(data, root=process)
MPI.COMM_WORLD.gather(data, root=process)
MPI.COMM_WORLD.Reduce(sendbuf, recvbuf, op=MPI.OPERATION, root=process)


# time
MPI.Wtime()

# no-buffer blocking
MPI.COMM_WORLD.send(data, dest=process, tag=number)
data = MPI.COMM_WORLD.recv(source=process, tag=number)

# no-buffer no-blocking
request = MPI.COMM_WORLD.isend(data, dest=process, tag=number)
request.wait()
request = MPI.COMM_WORLD.irecv(source=process, tag=number)
request.wait()

# buffer blocking with explicitly type
buffer = np.empty((), dtype=np.intc)
MPI.COMM_WORLD.Send([buffer, MPI.data_type], dest=process, tag=number)
MPI.COMM_WORLD.Recv([buffer, MPI.data_type], source=process, tag=number)