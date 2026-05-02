#include <iostream>
#include <cstdlib>
#include <mpi.h>

using namespace std;

bool isPrime(long long n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    for (long long i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }

    return true;
}

long long countPrimes(long long start, long long end) {
    long long count = 0;

    for (long long i = start; i <= end; i++) {
        if (isPrime(i)) {
            count++;
        }
    }

    return count;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long globalStart = 1;
    long long globalEnd = 10000;

    if (argc >= 3) {
        globalStart = atoll(argv[1]);
        globalEnd = atoll(argv[2]);
    }

    long long totalNumbers = globalEnd - globalStart + 1;
    long long chunkSize = totalNumbers / size;
    long long remainder = totalNumbers % size;

    long long localStart;
    long long localEnd;

    if (rank < remainder) {
        localStart = globalStart + rank * (chunkSize + 1);
        localEnd = localStart + chunkSize;
    }
    else {
        localStart = globalStart + rank * chunkSize + remainder;
        localEnd = localStart + chunkSize - 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    long long localCount = countPrimes(localStart, localEnd);

    long long globalCount = 0;

    MPI_Reduce(
        &localCount,
        &globalCount,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    if (rank == 0) {
        cout << "Number of primes in range ["
            << globalStart << "; " << globalEnd << "] = "
            << globalCount << endl;

        cout << "Execution time: " << endTime - startTime << " s" << endl;
        cout << "Number of processes: " << size << endl;
    }

    MPI_Finalize();

    return 0;
}