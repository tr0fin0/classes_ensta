from mpi4py import MPI


PROCESS_MAX     = 10
INTERATION_MAX  = 25

def serialWhile() -> None:
    interation  = 0
    process     = 0
    value       = 0


    while(interation < INTERATION_MAX):

        if (process+1) == PROCESS_MAX:
            process = 0
            print('='*31)
        else:
            print(f'process {process:2.0f} send {value:4.0f} to rank {process+1:2.0f}')
            process     += 1
            value       += 1
            interation  += 1


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