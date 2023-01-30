from mpi4py import MPI

def parallelWhile(value: int):
    comm = MPI.COMM_WORLD   # instantice the communication world
    size = comm.Get_size()  # size of the communication world
    rank = comm.Get_rank()  # process ID ('rank')
    # multiple instances of this programming is running at the same time

    print(rank)
    if rank != 0:
        comm.send(value, dest=(rank+1))
        value = comm.recv(rank)
        print(f'process {rank:2.0f} send {value:4.0f} to rank {rank+1:2.0f}')
    else:
        print("="*20)

    MPI.Finalize()
def main():
    serialWhile()
    # parallelWhile(0)


main()