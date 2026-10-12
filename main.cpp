#include "sparse_allreduce.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <exception>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
using namespace sparse_allreduce;

// ================
// Helper Functions
// ================

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int parse_int(const char* text) {
    size_t consumed = 0;
    int value = stoi(text, &consumed);
    if (consumed != strlen(text)) {
        throw invalid_argument("Invalid integer argument");
    }
    return value;
}

double parse_double(const char* text) {
    size_t consumed = 0;
    double value = stod(text, &consumed);
    if (consumed != strlen(text)) {
        throw invalid_argument("Invalid numeric argument");
    }
    return value;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0) {
        if (iplace == argc - 1) {
            throw invalid_argument(string("Missing value for ") + option);
        }
        return parse_int(argv[iplace + 1]);
    }

    return default_value;
}

double find_double_arg(int argc, char** argv, const char* option, double default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0) {
        if (iplace == argc - 1) {
            throw invalid_argument(string("Missing value for ") + option);
        }
        return parse_double(argv[iplace + 1]);
    }

    return default_value;
}


vector<int> find_int_array_arg(int argc, char** argv, const char* option) {
    vector<int> result;
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        for (int i = iplace + 1; i < argc; ++i) {
            // Stop at the next option. "-<digit>" is a value, so negative degrees reach validation.
            if (argv[i][0] == '-' && !isdigit(static_cast<unsigned char>(argv[i][1]))) {
                break;
            }
            result.push_back(parse_int(argv[i]));
        }
    }

    if (result.empty()) {
        throw invalid_argument("No valid array provided");
    }

    return result;
}

// Returns true for independent per-rank patterns, false for a pattern shared by all ranks.
bool find_independent_pattern(int argc, char** argv, const char* option, bool default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace == argc - 1 && iplace >= 0) {
        throw invalid_argument("Missing sparsity pattern");
    }

    if (iplace >= 0 && iplace < argc - 1) {
        char* pattern = argv[iplace + 1];
        if (strcmp(pattern, "shared") == 0) {
            return false;
        } else if (strcmp(pattern, "independent") == 0) {
            return true;
        } else {
            throw invalid_argument("Invalid sparsity pattern");
        }
    }

    return default_value;
}

// ==============
// Main Function
// ==============

int run_allreduce(int argc, char** argv, int num_procs, int rank) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        cout << "Options:" << endl;
        cout << "-h: see this help" << endl;
        cout << "-l <number>: set the sparsity level in percentage, e.g. 99 or 99.5" << endl;
        cout << "-p <string>: shared (same nonzero pattern on every rank) or independent (one pattern per rank)" << endl;
        cout << "-n <int>: set the number of elements in the vector" << endl;
        cout << "-s <int>: set the seed of random number generator" << endl;
        cout << "-c <int> [<int>]: one or two positive layer degrees whose product equals the MPI process count" << endl;
        return 0;
    }

    // Initialize Sparse Vectors
    // Default values:
    // sparsity_level = 90%
    // pattern = shared
    // num_elts = 100000
    // random_seed = 0
    double sparsity_level = find_double_arg(argc, argv, "-l", 90.0) / 100.0;
    bool independent = find_independent_pattern(argc, argv, "-p", false);
    int num_elts = find_int_arg(argc, argv, "-n", 100000);
    int random_seed = find_int_arg(argc, argv, "-s", 0);

    // Network Configuration
    vector<int> network = find_int_array_arg(argc, argv, "-c");
    if (num_elts < 0 || !(sparsity_level >= 0.0 && sparsity_level <= 1.0)) {
        throw invalid_argument("Vector length must be nonnegative and sparsity must be between 0 and 100");
    }
    const vector<layer_t> layers = make_layers(network, num_procs);

    const sparse_vec vec = init_sparse_vec(num_elts, random_seed, sparsity_level, independent ? rank + 1 : 0);
    commit_idx_val_pair_type();

    // Inputs differ across ranks in independent mode, so report the mean.
    long long total_input_nnz = 0;
    long long local_input_nnz = vec.nnz;
    MPI_Reduce(&local_input_nnz, &total_input_nnz, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    const int warmup_iterations = 3;
    const int measured_iterations = 5;
    vector<double> sparse_times;
    vector<double> dense_times;
    sparse_times.reserve(measured_iterations);
    dense_times.reserve(measured_iterations);

    // Generate/convert input once and reuse the dense output buffer. 
    // Input setup, barriers, timing reductions, and correctness checks are excluded from timing.
    const vector<int> dense_vec = convert_dense(vec);
    vector<int> reduced_dense_vec(num_elts, 0);
    int result_nnz = 0;

    for (int iteration = 0; iteration < warmup_iterations + measured_iterations; ++iteration) {
        const bool measured = iteration >= warmup_iterations;

        MPI_Barrier(MPI_COMM_WORLD);
        const double sparse_start = measured ? MPI_Wtime() : 0.0;
        // Keep communicator setup and the full sparse operation inside the timer.
        // The input is never modified, so every iteration starts from the original vector.
        const sparse_vec result_vec = sparse_allreduce::allreduce(vec, layers);
        const double sparse_time = measured ? MPI_Wtime() - sparse_start : 0.0;
        result_nnz = result_vec.nnz;

        MPI_Barrier(MPI_COMM_WORLD);
        const double dense_start = measured ? MPI_Wtime() : 0.0;
        MPI_Allreduce(dense_vec.data(), reduced_dense_vec.data(), num_elts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        const double dense_time = measured ? MPI_Wtime() - dense_start : 0.0;

        // Check all ranks on every iteration, including warmups, before accepting a sample.
        int local_match = match_dense(result_vec, reduced_dense_vec) ? 1 : 0;
        int all_match = 0;
        MPI_Allreduce(&local_match, &all_match, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!all_match) {
            if (rank == 0) {
                cerr << "Results Do Not Match! Iteration " << iteration + 1 << endl;
            }
            free_idx_val_pair_type();
            return 1;
        }

        if (measured) {
            const double local_times[2] = {sparse_time, dense_time};
            double max_times[2] = {};
            MPI_Reduce(local_times, max_times, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                sparse_times.push_back(max_times[0]);
                dense_times.push_back(max_times[1]);
            }
        }
    }

    if (rank == 0) {
        sort(sparse_times.begin(), sparse_times.end());
        sort(dense_times.begin(), dense_times.end());
        cout << "Timing: median of " << measured_iterations << " per-iteration rank maxima after " << warmup_iterations << " warmups" << endl;
        // Preserve the existing labels for the analysis notebook.
        cout << "Sparse Allreduce Time: " << sparse_times[measured_iterations / 2] << "s" << endl;
        cout << "Dense Allreduce Time: " << dense_times[measured_iterations / 2] << "s" << endl;
        cout << "Sparse Allreduce Range: " << sparse_times.front() << "s - " << sparse_times.back() << "s" << endl;
        cout << "Dense Allreduce Range: " << dense_times.front() << "s - " << dense_times.back() << "s" << endl;
        cout << "Sparse Vector Length: " << num_elts << endl;
        cout << "Mean Input Non-Zero Elements: " << static_cast<double>(total_input_nnz) / num_procs << endl;
        // Nonzeros in the reduced result.
        cout << "Sparse Vector Non-Zero Elements: " << result_nnz << endl;
        cout << "Results Match!" << endl;
    }

    free_idx_val_pair_type();
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status = 1;
    try {
        status = run_allreduce(argc, argv, num_procs, rank);
    } catch (const exception& error) {
        cerr << "Rank " << rank << ": " << error.what() << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
