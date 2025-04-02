#include <bits/stdc++.h>
#include <fstream>
#include <mpi.h>

#pragma GCC optimize("O3")
#pragma loop_opt(on)

using namespace std;

using LL = long long;
#define INF 50000
#define pb push_back

struct Edge {
    int x, y, c;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int n;
    vector<Edge> edges;
    int num_edges;

    if (myid == 0) {
        string file_name;
        cin >> file_name;
        ifstream file(file_name);
        if (!file.is_open()) {
            cerr << "Could not open the file " << file_name << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        file >> n; 
        int x, y, c;
        while (file >> x >> y >> c) {
            edges.pb({x, y, c});
        }

        num_edges = edges.size();
        file.close();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int edges_per_proc = num_edges / numprocs;
    int remainder = num_edges % numprocs;

    vector<int> dist(n, INF);
    dist[0] = 0;

    vector<Edge> edges_send(num_edges);
    if (myid == 0) {
        //sendbuff
        edges_send = edges;
    }

    vector<int> sendcounts(numprocs, 0);
    for (int i = 0; i < numprocs; i++) {
        sendcounts[i] = edges_per_proc * sizeof(Edge);
    }
    sendcounts[numprocs-1] += remainder * sizeof(Edge);

    vector<int> displs(numprocs, 0);
    for (int i = 1; i < numprocs; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    //recvbuf
    vector<Edge> local_edges_recv(sendcounts[myid] / sizeof(Edge));

    MPI_Scatterv(edges_send.data(), sendcounts.data(), displs.data(), MPI_BYTE,
                  local_edges_recv.data(), sendcounts[myid], MPI_BYTE,
                  0, MPI_COMM_WORLD);

    int j;
    int min_distance;
    int proc;
    int idx;
    bool updated;
    bool global_updated = true;
    vector<int> gather_buffer(n);

    // Bellman-Ford algorithm
    for (int i = 0; i < n; i++) {
        updated = false;

        for (const auto& edge : local_edges_recv) {
            if (dist[edge.x] != INF && dist[edge.x] + edge.c < dist[edge.y]) {
                dist[edge.y] = dist[edge.x] + edge.c;
                updated = true;
            }
        }

        MPI_Allreduce(&updated, &global_updated, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        if (!global_updated) {
            break;
        }

        MPI_Request request;
        MPI_Ireduce(dist.data(), gather_buffer.data(), n, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD, &request);
        if (myid == 0) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            dist = gather_buffer;
        }

        MPI_Ibcast(dist.data(), n, MPI_INT, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    if (myid == 0) {
        for (int i = 0; i < n; i++) {
            cout << dist[i] << ' ';
        }
    }

    MPI_Finalize();
    return 0;
}