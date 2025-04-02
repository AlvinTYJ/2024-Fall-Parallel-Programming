#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int wrapIndex(int x, int max) {
    if (x < 0) {
        return max + (x % max);
    }
    if (x >= max) {
        return x % max;
    }
    return x;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int t, n, m;
    char file_name[50];
    FILE *file;

    if (myid == 0) {
        scanf("%s", file_name);
        file = fopen(file_name, "r");
        if (file == NULL) {
            printf("Error opening file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf(file, "%d", &t);
        fscanf(file, "%d %d", &n, &m);
    }

    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *A[2];
    for (int i = 0; i < 2; i++) {
        A[i] = malloc(n * m * sizeof(int));
    }

    int D;
    int K[3][3];

    if (myid == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                fscanf(file, "%d", &A[0][i * m + j]);
            }
        }

        fscanf(file, "%d", &D);
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                fscanf(file, "%d", &K[i][j]);
            }
        }
        fclose(file);
    }

    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(K, 3 * 3, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = n / numprocs;
    int extra_rows = n % numprocs;
    int start_row = myid * rows_per_proc + (myid < extra_rows ? myid : extra_rows);
    int end_row = start_row + rows_per_proc + (myid < extra_rows ? 1 : 0);

    int i, j, row, sum, di, dj, ni, nj, rows;

    int *sendcounts = malloc(numprocs * sizeof(int));
    int *displs = malloc(numprocs * sizeof(int));
    int *local_data = malloc((end_row - start_row) * m * sizeof(int));

    for (int time = 0; time < t; time++) {
        MPI_Bcast(A[time % 2], n * m, MPI_INT, 0, MPI_COMM_WORLD);

        for (row = start_row; row < end_row; row++) {
            for (j = 0; j < m; j++) {
                sum = 0;
                for (di = -1; di <= 1; di++) {
                    for (dj = -1; dj <= 1; dj++) {
                        ni = wrapIndex(row + di, n);
                        nj = wrapIndex(j + dj, m);
                        sum += K[1 + di][1 + dj] * A[time % 2][ni * m + nj];
                    }
                }
                A[(time + 1) % 2][row * m + j] = sum / (D * D);
            }
        }

        for (int i = 0; i < numprocs; i++) {
            rows = n / numprocs + (i < extra_rows ? 1 : 0);
            sendcounts[i] = rows * m;
            displs[i] = (i * rows_per_proc + (i < extra_rows ? i : extra_rows)) * m;
        }

        for (int i = start_row; i < end_row; i++) {
            for (j = 0; j < m; j++) {
                local_data[(i - start_row) * m + j] = A[(time + 1) % 2][i * m + j];
            }
        }

        MPI_Gatherv(local_data, (end_row - start_row) * m, MPI_INT,
                    A[(time + 1) % 2], sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (myid == 0) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                printf("%d ", A[t % 2][i * m + j]);
            }
        }
    }

    MPI_Finalize();
    return 0;
}