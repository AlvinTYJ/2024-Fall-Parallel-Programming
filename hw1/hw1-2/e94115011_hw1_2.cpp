#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <utility>
#include <fstream>
#include <mpi.h>
using namespace std;

struct Point {
    int x, y, index;
    bool operator<(const Point& other) const {
        return (x < other.x) || (x == other.x && y < other.y);
    }
};

long long cross(const Point& O, const Point& A, const Point& B) {
    long long val = (long long)(A.x - O.x) * (B.y - O.y) - (long long)(A.y - O.y) * (B.x - O.x);
    return val;
}

vector<Point> computeConvexHull(vector<Point>& pts, vector<Point>& upper, vector<Point>& lower) {
    if (pts.size() <= 1)
        return pts;

    sort(pts.begin(), pts.end());

    upper.clear();
    lower.clear();
    upper.reserve(pts.size());
    lower.reserve(pts.size());

    for (const auto& p : pts) {
        while (upper.size() >= 2 && cross(upper[upper.size()-2], upper[upper.size()-1], p) >= 0)
            upper.pop_back();
        upper.push_back(p);
    }

    for (int i = pts.size()-1; i >= 0; i--) {
        const auto& p = pts[i];
        while (lower.size() >= 2 && cross(lower[lower.size()-2], lower[lower.size()-1], p) >= 0)
            lower.pop_back();
        lower.push_back(p);
    }

    if (!upper.empty()) upper.pop_back();
    if (!lower.empty()) lower.pop_back();
    upper.insert(upper.end(), lower.begin(), lower.end());
    return upper;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int n;
    vector<Point> all_points;

    if (myid == 0) {
        string file_name;
        cin >> file_name;
        ifstream file(file_name);
        if (!file.is_open()) {
            cerr << "Could not open the file " << file_name << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        file >> n;
        all_points.resize(n);
        for (int i = 0; i < n; i++) {
            file >> all_points[i].x >> all_points[i].y;
            all_points[i].index = i + 1;
        }
        file.close();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int pointsChunkSize = n / numprocs;
    int remainder = n % numprocs;

    vector<int> sendcounts(numprocs, 0);
    for (int i = 0; i < numprocs; i++) {
        sendcounts[i] = pointsChunkSize * 2 + (i < remainder ? 2 : 0);
    }

    vector<int> displs(numprocs, 0);
    for(int i = 1; i < numprocs; i++) {
        displs[i] = displs[i-1] + sendcounts[i-1];
    }

    int recv_count = sendcounts[myid];
    vector<int> recv_buffer(recv_count);

    vector<int> send_buffer;
    if(myid ==0) {
        send_buffer.reserve(n * 2);
        for(auto &p : all_points) {
            send_buffer.push_back(p.x);
            send_buffer.push_back(p.y);
        }
    }

    MPI_Scatterv(send_buffer.data(), sendcounts.data(), displs.data(), MPI_INT,
                 recv_buffer.data(), recv_count, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = recv_count / 2;
    vector<Point> local_points(local_n);
    for(int i = 0; i < local_n; i++) {
        local_points.emplace_back(Point{recv_buffer[2*i], recv_buffer[2*i+1], 0});
    }

    vector<Point> upper, lower;
    vector<Point> local_hull = computeConvexHull(local_points, upper, lower);

    int local_send_count = local_hull.size() * 2;
    vector<int> hull_sizes(numprocs);

    MPI_Gather(&local_send_count, 1, MPI_INT, hull_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> hull_displs;
    vector<int> gather_buffer;
    if(myid == 0) {
        hull_displs.resize(numprocs, 0);
        int total = 0;
        for(int i =0; i < numprocs; i++) {
            hull_displs[i] = total;
            total += hull_sizes[i];
        }
        gather_buffer.resize(total);
    }

    vector<int> local_hull_buffer(local_send_count);
    for(int i = 0; i < local_hull.size(); i++) {
        local_hull_buffer[2*i] = local_hull[i].x;
        local_hull_buffer[2*i+1] = local_hull[i].y;
    }

    MPI_Gatherv(local_hull_buffer.data(), local_send_count, MPI_INT,
                gather_buffer.data(), hull_sizes.data(), hull_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if(myid ==0) {
        vector<Point> final_points;
        final_points.reserve(n);

        for(int i=0; i<numprocs; i++) {
            int num_points = hull_sizes[i] / 2;
            for(int j=0; j < num_points; j++) {
                final_points.emplace_back(Point{
                    gather_buffer[hull_displs[i] + 2*j],
                    gather_buffer[hull_displs[i] + 2*j +1],
                    0
                });
            }
        }

        sort(final_points.begin(), final_points.end());
        final_points.erase(unique(final_points.begin(), final_points.end(),
        [](const Point& a, const Point& b) { return a.x == b.x && a.y == b.y; }),
        final_points.end());

        vector<Point> global_hull = computeConvexHull(final_points, upper, lower);

        map<pair<int, int>, int> point_map;
        for(auto &p : all_points) {
            point_map[{p.x, p.y}] = p.index;
        }
        for(auto &p : global_hull) {
            p.index = point_map[{p.x, p.y}];
        }

        int start = 0;
        for(int i=1;i<global_hull.size();i++) {
            if( (global_hull[i].x < global_hull[start].x) ||
                (global_hull[i].x == global_hull[start].x && global_hull[i].y < global_hull[start].y) )
                start = i;
        }

        rotate(global_hull.begin(), global_hull.begin()+start, global_hull.end());

        for(int i=0;i<global_hull.size();i++) {
            cout << global_hull[i].index << " ";
        }
    }

    MPI_Finalize();
    return 0;
}