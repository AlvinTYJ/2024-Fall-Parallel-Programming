#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int n, m;
    int myid, numprocs, i, valid_combinations = 0, global_valid_combinations = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int tests[32][32] = {0};
    int parts_count[32];
    int cost[32];

    if (myid == 0) {
        char file_name[50];
        scanf("%s", file_name);
        FILE *file = fopen(file_name, "r");
        if (file == NULL) {
            printf("Could not open the file %s\n", file_name);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        fscanf(file, "%d %d", &n, &m);
        for (int i = 0; i < m; i++) {
            fscanf(file, "%d %d", &parts_count[i], &cost[i]);
            for (int j = 0; j < parts_count[i]; j++) {
                fscanf(file, "%d", &tests[i][j]);
            }
        }

        fclose(file);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(parts_count, m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cost, m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tests, 32 * 32, MPI_INT, 0, MPI_COMM_WORLD);

    int total_program_parts = (1 << n) - 1;
    int subset_size = (1 << m);
    int chunk_size = subset_size / numprocs;
    int start = myid * chunk_size;
    int end = (myid == numprocs - 1) ? subset_size : start + chunk_size;

    for (i = start; i < end; i++) {
        int covered_parts = 0;
        for (int j = 0; j < m; j++) {
            if (i & (1 << j)) {
                for (int k = 0; k < parts_count[j]; k++) {
                    covered_parts |= (1 << (tests[j][k] - 1));
                }
            }
        }

        if (covered_parts == total_program_parts) {
            valid_combinations++;
        }
    }

    MPI_Reduce(&valid_combinations, &global_valid_combinations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("%d", global_valid_combinations);
    }
    MPI_Finalize();
    return 0;
}