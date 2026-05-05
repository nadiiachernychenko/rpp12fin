#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <vector>
#include <fstream>

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

vector<long long> findPrimes(long long start, long long end) {
    vector<long long> primes;

    for (long long i = start; i <= end; i++) {
        if (isPrime(i)) {
            primes.push_back(i);
        }
    }

    return primes;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long globalStart = 1;
    long long globalEnd = 100;

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

    vector<long long> localPrimes = findPrimes(localStart, localEnd);

    int localCount = static_cast<int>(localPrimes.size());

    vector<int> recvCounts(size);
    vector<int> displs(size);

    MPI_Gather(
        &localCount,
        1,
        MPI_INT,
        recvCounts.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    vector<long long> allPrimes;

    if (rank == 0) {
        int totalPrimeCount = 0;

        for (int i = 0; i < size; i++) {
            displs[i] = totalPrimeCount;
            totalPrimeCount += recvCounts[i];
        }

        allPrimes.resize(totalPrimeCount);
    }

    MPI_Gatherv(
        localPrimes.data(),
        localCount,
        MPI_LONG_LONG,
        allPrimes.data(),
        recvCounts.data(),
        displs.data(),
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    if (rank == 0) {
        ofstream file("primes.txt");

        if (file.is_open()) {
            for (long long prime : allPrimes) {
                file << prime << endl;
            }

            file.close();

            cout << "Number of primes in range ["
                << globalStart << "; " << globalEnd << "] = "
                << allPrimes.size() << endl;

            cout << "All primes were written to primes.txt" << endl;
        }
        else {
            cout << "Error: cannot open file primes.txt" << endl;
        }

        cout << "Execution time: " << endTime - startTime << " s" << endl;
        cout << "Number of processes: " << size << endl;
    }

    MPI_Finalize();

    return 0;
}