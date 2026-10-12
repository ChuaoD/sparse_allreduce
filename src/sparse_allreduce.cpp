#include "sparse_allreduce.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mpi.h>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace std;

namespace sparse_allreduce {

MPI_Datatype IDX_VAL_PAIR;

// ===============
// Vector Routines
// ===============

// Initialize Sparse Vector
sparse_vec init_sparse_vec(int num_elts, int random_seed, double sparsity_level, int stream) {
    if (num_elts < 0 || !isfinite(sparsity_level) || sparsity_level < 0.0 || sparsity_level > 1.0 || stream < 0) {
        throw invalid_argument("Invalid sparse vector parameters");
    }
    sparse_vec vec;
    vec.length = num_elts;
    vec.nnz= 0;
    if (num_elts == 0 || sparsity_level == 1.0) {
        return vec;
    }
    random_device rd;
    const unsigned int seed = random_seed ? static_cast<unsigned int>(random_seed) : rd();
    mt19937 gen(seed);
    if (stream != 0) {
        // Mix the stream into the seed so neighbouring ranks do not get correlated generators.
        seed_seq stream_seed{seed, static_cast<unsigned int>(stream)};
        gen.seed(stream_seed);
    }

    // Visit indices in order, so the data is already sorted.
    uniform_int_distribution<int> unif_dist(0, num_elts - 1);
    for (int i = 0; i < num_elts; ++i) {
        if ((double) unif_dist(gen) / num_elts >= sparsity_level) {
            vec.data.push_back({i, 1});
            vec.nnz++;
        }
    }

    return vec;
}

// Convert Sparse Vector of Index-Value Pairs to Dense Vector Representation
vector<int> convert_dense(const sparse_vec& vec) {
    vector<int> dense_vec(vec.length, 0);
    for (const idx_val_pair& pair : vec.data) {
        dense_vec[pair.idx] = pair.val;
    }
    return dense_vec;
}

// Check the values and the sorted, unique, nonzero sparse representation.
bool match_dense(const sparse_vec& sparse, const vector<int>& dense) {
    if (sparse.length != static_cast<int>(dense.size()) || sparse.nnz != static_cast<int>(sparse.data.size())) {
        return false;
    }
    size_t next = 0;
    for (size_t i = 0; i < dense.size(); ++i) {
        int value = 0;
        if (next < sparse.data.size() && sparse.data[next].idx == static_cast<int>(i)) {
            value = sparse.data[next++].val;
            if (value == 0) {
                return false;
            }
        }
        if (value != dense[i]) {
            return false;
        }
    }
    return next == sparse.data.size();
}

// =====================
// Network Configuration
// =====================

// Build the Layers of the Communication Network
vector<layer_t> make_layers(const vector<int>& degrees, int num_procs) {
    if (degrees.empty()) {
        throw invalid_argument("At least one layer degree is required");
    }
    if (degrees.size() > 2) {
        throw invalid_argument("Only one- and two-layer networks are supported");
    }
    int remaining_procs = num_procs;
    for (int degree : degrees) {
        if (degree <= 0 || remaining_procs % degree != 0) {
            throw invalid_argument("Layer degrees must be positive and their product must equal the MPI process count");
        }
        remaining_procs /= degree;
    }
    if (remaining_procs != 1) {
        throw invalid_argument("Layer degree product must equal the MPI process count");
    }

    vector<layer_t> layers(degrees.size());
    layers[0].in_degree = 1;
    layers[0].out_degree = degrees[0];
    layers[0].depth = 1;
    for (size_t i = 1; i < degrees.size(); ++i) {
        layers[i].in_degree = degrees[i - 1];
        layers[i].out_degree = degrees[i];
        layers[i].depth = static_cast<int>(i) + 1;
    }
    return layers;
}

// ============
// MPI Datatype
// ============

// Create MPI Datatype for Index-Value Pair
void commit_idx_val_pair_type() {
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    int block_lengths[2] = {1, 1};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(idx_val_pair, idx);
    offsets[1] = offsetof(idx_val_pair, val);
    MPI_Type_create_struct(2, block_lengths, offsets, types, &IDX_VAL_PAIR);
    MPI_Type_commit(&IDX_VAL_PAIR);
}

void free_idx_val_pair_type() {
    MPI_Type_free(&IDX_VAL_PAIR);
}

// ============
// MPI Routines
// ============

// Split the MPI_COMM_WORLD into Groups
void split_communicator(layer_t layer, int rank, MPI_Comm& group_comm, int& in_group_rank) {
    int color;
    if (layer.depth == 1) {
        color = rank / layer.out_degree;
    } else {
        color = in_group_rank;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &group_comm);
    MPI_Comm_rank(group_comm, &in_group_rank);
}

// Calculate the Count of Elements to Scatter in the Group
void calc_scatter_counts(const sparse_vec& vec, layer_t layer, vector<int>& counts) {

    fill(counts.begin(), counts.end(), 0);
    // Split global index ranges consistently, including uneven and empty ranges.
    // At depth two, each communicator owns one first-layer range; modulo selects
    // its subrange without inferring the range start from the first nonzero.
    const int64_t partitions = static_cast<int64_t>(layer.in_degree) * layer.out_degree;
    for (const idx_val_pair& pair : vec.data) {
        const int destination = (static_cast<int64_t>(pair.idx) * partitions / vec.length) % layer.out_degree;
        counts[destination]++;
    }
}

// Calculate the Count of Elements to Receive in the Group
int calc_recv_counts(const vector<int>& send_counts, vector<int>& recv_counts, vector<int>& send_dipls, vector<int>& recv_dipls, MPI_Comm group_comm) {
     MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, group_comm);

    // Compute the Total Number of Elements to Receive and Displacements
    const int64_t total = accumulate(recv_counts.begin(), recv_counts.end(), int64_t{0});
    if (total > numeric_limits<int>::max()) {
        throw overflow_error("Received sparse data exceeds MPI int count capacity");
    }
    int total_recv_count = static_cast<int>(total);
    for (size_t i = 1; i < send_dipls.size(); ++i) {
        send_dipls[i] = send_dipls[i - 1] + send_counts[i - 1];
        recv_dipls[i] = recv_dipls[i - 1] + recv_counts[i - 1];
    }
    return total_recv_count;
}

// Scatter Sparse Vector to Each Process in the Group
void scatter_sparse_vec(const sparse_vec& vec, const vector<int>& send_counts, const vector<int>& send_dipls, const vector<int>& recv_counts, const vector<int>& recv_dipls, MPI_Comm group_comm, vector<idx_val_pair>& recv_data) {

    MPI_Alltoallv(vec.data.data(), send_counts.data(), send_dipls.data(), IDX_VAL_PAIR, recv_data.data(), recv_counts.data(), recv_dipls.data(), IDX_VAL_PAIR, group_comm);
}

// Reduce Sparse Vector for Each Process
void reduce_sparse_vec(const vector<idx_val_pair>& vec_data, const vector<int>& counts, vector<idx_val_pair>& reduced_data) {
    // Keep one cursor per sorted input segment, including empty segments.
    typedef pair<int, size_t> heap_entry; // (vector index, segment number)
    priority_queue<heap_entry, vector<heap_entry>, greater<heap_entry>> pq;
    vector<size_t> next(counts.size());
    vector<size_t> ends(counts.size());
    size_t offset = 0;
    for (size_t part = 0; part < counts.size(); ++part) {
        if (counts[part] < 0 || static_cast<size_t>(counts[part]) > vec_data.size() - offset) {
            throw invalid_argument("Invalid sparse merge counts");
        }
        next[part] = offset;
        offset += static_cast<size_t>(counts[part]);
        ends[part] = offset;
        if (next[part] < ends[part]) {
            pq.push({vec_data[next[part]].idx, part});
        }
    }
    if (offset != vec_data.size()) {
        throw invalid_argument("Sparse merge counts do not cover the input");
    }

    reduced_data.clear();
    while (!pq.empty()) {
        const int idx = pq.top().first;
        int64_t value = 0;
        do {
            const size_t part = pq.top().second;
            pq.pop();
            value += vec_data[next[part]].val;
            ++next[part];
            if (next[part] < ends[part]) {
                pq.push({vec_data[next[part]].idx, part});
            }
        } while (!pq.empty() && pq.top().first == idx);
        if (value < numeric_limits<int>::min() || value > numeric_limits<int>::max()) {
            throw overflow_error("Sparse sum exceeds int capacity");
        }
        if (value != 0) {
            reduced_data.push_back({idx, static_cast<int>(value)});
        }
    }
}

// Scatter-Reduce Sparse Vector
sparse_vec scatter_reduce_sparse_vec(int num_elts, const sparse_vec& vec, layer_t layer, MPI_Comm group_comm, PhaseTimers* timers) {

    const double scatter_start = timers ? MPI_Wtime() : 0.0;
    // Calculate the Counts for Scatter in Each Group
    vector<int> send_counts(layer.out_degree, 0);
    calc_scatter_counts(vec, layer, send_counts);

    // Calculate the Counts for Receive in Each Group
    vector<int> recv_counts(layer.out_degree, 0);
    vector<int> send_dipls(layer.out_degree, 0);
    vector<int> recv_dipls(layer.out_degree, 0);
    int total_recv_count = calc_recv_counts(send_counts, recv_counts, send_dipls, recv_dipls, group_comm);

    // Scatter the Sparse Vector to Each Process inside the Group
    vector<idx_val_pair> recv_data(total_recv_count);
    scatter_sparse_vec(vec, send_counts, send_dipls, recv_counts, recv_dipls, group_comm, recv_data);
    if (timers) {
        timers->scatter += MPI_Wtime() - scatter_start;
    }

    const double reduce_start = timers ? MPI_Wtime() : 0.0;
    // Reduce the Sparse Vector for Each Process
    vector<idx_val_pair> reduced_data;
    reduce_sparse_vec(recv_data, recv_counts, reduced_data);
    if (timers) {
        timers->reduce += MPI_Wtime() - reduce_start;
    }

    sparse_vec reduced_vec;
    reduced_vec.length = num_elts;
    reduced_vec.nnz = reduced_data.size();
    reduced_vec.data = reduced_data;

    return reduced_vec;
}

// Sparse Allreduce over MPI_COMM_WORLD
sparse_vec allreduce(const sparse_vec& vec, const vector<layer_t>& layers, PhaseTimers* timers) {
    if (layers.empty()) {
        throw invalid_argument("At least one layer is required");
    }
    int rank = 0;
    int num_procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    const int num_elts = vec.length;

    // The first layer reads the caller's input directly; later layers read the previous result.
    sparse_vec reduced_vec = {num_elts, 0, {}};
    const sparse_vec* layer_input = &vec;
    int in_group_rank = rank;
    for (const layer_t& layer : layers) {
        MPI_Comm group_comm;
        const double split_start = timers ? MPI_Wtime() : 0.0;
        split_communicator(layer, rank, group_comm, in_group_rank);
        if (timers) {
            timers->split += MPI_Wtime() - split_start;
        }
        reduced_vec = scatter_reduce_sparse_vec(num_elts, *layer_input, layer, group_comm, timers);
        layer_input = &reduced_vec;
        MPI_Comm_free(&group_comm);
    }

    // Gather Every Rank's Reduced Segment
    const double all_gather_start = timers ? MPI_Wtime() : 0.0;
    vector<int> nnz_counts(num_procs, 0);
    MPI_Allgather(&reduced_vec.nnz, 1, MPI_INT, nnz_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    const int64_t gathered_count = accumulate(nnz_counts.begin(), nnz_counts.end(), int64_t{0});
    if (gathered_count > numeric_limits<int>::max()) {
        throw overflow_error("Gathered sparse data exceeds MPI int count capacity");
    }
    const int total_nnz = static_cast<int>(gathered_count);

    vector<idx_val_pair> all_data(total_nnz);
    vector<int> displs(num_procs, 0);
    for (int i = 1; i < num_procs; ++i) {
        displs[i] = displs[i - 1] + nnz_counts[i - 1];
    }
    MPI_Allgatherv(reduced_vec.data.data(), reduced_vec.nnz, IDX_VAL_PAIR, all_data.data(), nnz_counts.data(), displs.data(), IDX_VAL_PAIR, MPI_COMM_WORLD);
    if (timers) {
        timers->allgather += MPI_Wtime() - all_gather_start;
    }

    // One layer gathers ranges in global order. After two layers, each rank still owns one
    // disjoint index range, but rank order differs from range order, so reorder the segments
    // by their first index instead of merging them.
    const double reduce_start = timers ? MPI_Wtime() : 0.0;
    vector<idx_val_pair> result_data;
    if (layers.size() == 1) {
        result_data = std::move(all_data);
    } else {
        vector<int> segments;
        for (int i = 0; i < num_procs; ++i) {
            if (nnz_counts[i] > 0) {
                segments.push_back(i);
            }
        }
        sort(segments.begin(), segments.end(), [&](int a, int b) {
            return all_data[displs[a]].idx < all_data[displs[b]].idx;
        });
        result_data.reserve(all_data.size());
        for (int segment : segments) {
            if (!result_data.empty() && all_data[displs[segment]].idx <= result_data.back().idx) {
                throw logic_error("Reduced segments overlap");
            }
            result_data.insert(result_data.end(), all_data.begin() + displs[segment], all_data.begin() + displs[segment] + nnz_counts[segment]);
        }
    }
    const int result_nnz = static_cast<int>(result_data.size());
    sparse_vec result_vec = {num_elts, result_nnz, std::move(result_data)};
    if (timers) {
        timers->reduce += MPI_Wtime() - reduce_start;
    }
    return result_vec;
}

} // namespace sparse_allreduce
