#ifndef SPARSE_ALLREDUCE_H
#define SPARSE_ALLREDUCE_H

#include <mpi.h>
#include <vector>

namespace sparse_allreduce {

// Index-Value Pair Structure for Sparse Vectors
struct idx_val_pair {
    int idx;
    int val;
};

// IDX_VAL_PAIR and communication volume assume a packed pair of 32-bit ints.
static_assert(sizeof(idx_val_pair) == 8, "idx_val_pair must be 8 bytes");

// Sparse Vector Structure
struct sparse_vec {
    int length; // Number of elements in the vector
    int nnz; // Number of non-zero elements in the vector
    std::vector<idx_val_pair> data;
};

// Layer Structure
struct layer_t {
    int in_degree; // Group size of the input group
    int out_degree; // Group size of the output group
    int depth; // Depth in the network
};

// Per-phase timers. Passing nullptr skips all phase timing.
struct PhaseTimers {
    double split = 0.0; // Communicator creation
    double scatter = 0.0; // Count exchange and Alltoallv
    double reduce = 0.0; // Local merges and the final two-layer segment reordering
    double allgather = 0.0; // Result count exchange and Allgatherv
};

// Vector Routines
// Draw a sorted sparse vector with uniformly placed nonzeros. A nonzero seed makes the
// pattern repeatable; ranks passing the same seed and stream get identical patterns,
// while distinct streams (e.g. rank + 1) give independent patterns. Stream 0 seeds
// mt19937 with the seed directly.
sparse_vec init_sparse_vec(int num_elts, int random_seed, double sparsity_level, int stream = 0);

std::vector<int> convert_dense(const sparse_vec& vec);

// Check the values and the sorted, unique, nonzero sparse representation.
bool match_dense(const sparse_vec& sparse, const std::vector<int>& dense);

// Network Configuration
// Build one or two layers from degrees whose product equals num_procs; throws otherwise.
std::vector<layer_t> make_layers(const std::vector<int>& degrees, int num_procs);

// MPI Datatype for idx_val_pair; commit before calling allreduce and free before MPI_Finalize.
extern MPI_Datatype IDX_VAL_PAIR;

void commit_idx_val_pair_type();

void free_idx_val_pair_type();

// MPI Routines
void split_communicator(layer_t layer, int rank, MPI_Comm& group_comm, int& in_group_rank);

void calc_scatter_counts(const sparse_vec& vec, layer_t layer, std::vector<int>& counts);

int calc_recv_counts(const std::vector<int>& send_counts, std::vector<int>& recv_counts, std::vector<int>& send_dipls, std::vector<int>& recv_dipls, MPI_Comm group_comm);

void scatter_sparse_vec(const sparse_vec& vec, const std::vector<int>& send_counts, const std::vector<int>& send_dipls, const std::vector<int>& recv_counts, const std::vector<int>& recv_dipls, MPI_Comm group_comm, std::vector<idx_val_pair>& recv_data);

void reduce_sparse_vec(const std::vector<idx_val_pair>& vec_data, const std::vector<int>& counts, std::vector<idx_val_pair>& reduced_data);

sparse_vec scatter_reduce_sparse_vec(int num_elts, const sparse_vec& vec, layer_t layer, MPI_Comm group_comm, PhaseTimers* timers = nullptr);

// Sum every rank's sorted sparse vector over MPI_COMM_WORLD. Every rank receives the
// sorted sum. The input is only read, so repeated calls start from the same data.
sparse_vec allreduce(const sparse_vec& vec, const std::vector<layer_t>& layers, PhaseTimers* timers = nullptr);

} // namespace sparse_allreduce

#endif
