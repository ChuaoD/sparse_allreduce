# Sparse Allreduce in C++ and MPI

Allreduce adds up a vector across all processes and gives every process a copy of the total. When the vectors are mostly zeros, sending all those zeros seems wasteful. This project implements an Allreduce that sends only the nonzero entries, as (index, value) pairs, and measures when that actually beats the standard dense `MPI_Allreduce`.

The answer depends mostly on one thing: whether the processes (MPI ranks) have their nonzeros in the same places. I call this the nonzero **pattern**. Every experiment runs twice:

- **Shared pattern:** all ranks have nonzeros at the same indices. This is the best case for the sparse version, because the sum is no denser than any one input.
- **Independent patterns:** each rank picks its own indices. The sum then **fills in**: each rank adds nonzeros where the others had zeros, so the result is much denser than the inputs.

Key results, with 64 ranks, each holding a vector of length 5 million with 99% of its entries zero:

- With a shared pattern, the sparse version is **10.4× faster** than `MPI_Allreduce` with a one-layer network and **13.2× faster** with a two-layer (16 × 4) network.
- With independent patterns, it is only **1.15× faster**, basically a tie, because the sum ends up about 47% nonzero.

[Read the report](report.pdf) · [Explore the analysis notebook](scaling/analysis.ipynb) · [Inspect the measurements](scaling/)

## Highlights

- **Parallel implementation.** Sparse (index, value) storage with a custom MPI datatype, communicators split into groups for each network layer, and variable-size exchanges with `MPI_Alltoallv` and `MPI_Allgatherv`.
- **Merging with a heap.** Each rank receives several sorted lists of pairs and combines them with a k-way heap merge, adding up pairs that share an index. The output comes out sorted, so no final sort is needed, and the heap never holds more than one entry per list.
- **A speedup found by profiling.** With independent patterns and a vector length of 1 million per rank, the two-layer network was 2–3× slower than one layer. Its extra time grew with the size of the result (half the pairs, about half the extra time), which pointed to a step that does work for every pair: an unnecessary `O(n log p)` heap merge at the end. Each rank's block covers its own index range, so putting the blocks in order is enough. That change made the two-layer network about 45% faster.
- **Correctness checks.** Every run compares the sparse result with `MPI_Allreduce` on every rank, in every iteration. The code also checks the network shape, MPI count limits, overflow in sums, and that reordered blocks don't overlap.
- **Honest reporting.** The results show where the sparse version loses as well as where it wins, and the limits of the measurements are spelled out. All raw logs and the analysis notebook are included.

## Results

All 1,032 runs passed the correctness check.

### Shared vs. independent patterns

Speedup is dense time divided by sparse time, so anything above 1 means the sparse version is faster. With 64 ranks and a vector length of 5 million per rank (zeros included):

| Pattern | Network | 90% | 95% | 98% | 99% | 99.5% | 99.9% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared | One layer | 1.27× | 2.53× | 5.60× | 10.40× | 16.02× | 44.21× |
| Shared | 16 × 4 | 1.66× | 3.00× | 7.24× | 13.16× | 22.15× | 55.34× |
| Independent | One layer | 0.46× | 0.55× | 0.74× | 1.15× | 2.14× | 8.54× |
| Independent | 16 × 4 | 0.33× | 0.38× | 0.55× | 0.85× | 1.48× | 5.60× |

With a vector length of 1 million per rank, across all 18 rank counts tested:

- **Shared pattern:** sparse wins at every rank count from 95% sparsity up, and at 14 of 18 rank counts at 90%.
- **Independent patterns:** sparse wins at every rank count only from 99.5% up. At 99% it wins at 16 of 18 rank counts, at 98% at 6, at 95% at 1, and at 90% at none.

Logs: [shared](scaling/shared/) and [independent](scaling/independent/).

### When does sparse start to win?

With independent patterns, each rank's nonzeros land in new places, so the sum gets denser as ranks are added. If a fraction `q` of each rank's entries is nonzero, the sum over `p` ranks is about `1 − (1 − q)^p` nonzero. This formula matches all 108 independent-pattern runs in the rank-count sweep to within 0.07 percentage points.

I call the sparsity at which sparse and dense take the same time the **crossover**. Since only six sparsity levels were tested, I report it as a range: between the last level where sparse loses and the first level where it wins. With a vector length of 1 million per rank and a one-layer network:

| Ranks | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shared | 90–95% | 90–95% | 90–95% | 90–95% | ≤ 90% | ≤ 90% | ≤ 90% |
| Independent | 95–98% | 95–98% | 98–99% | 98–99% | 98–99%* | 99–99.5% | 99–99.5% |

"≤ 90%" means sparse already wins at 90%, the lowest level tested. \*At 32 ranks, sparse wins at 99% by only 1.04×, which is really a tie; the same goes for 72 to 112 ranks (1.02–1.08×). The [notebook](scaling/analysis.ipynb) has the full table for all 18 rank counts.

**Why sparse loses with independent patterns.** The last step of the sparse version, the Allgather, sends every rank a copy of the whole summed vector. At 64 ranks, a vector length of 1 million, and 99% sparsity, that vector is about 47% nonzero with independent patterns but only 1% with a shared pattern, so each rank receives about 475,000 pairs instead of about 10,000. The Allgather alone then takes 6.8 ms of the 8.2 ms total (compared with 0.6 ms of 1.8 ms with a shared pattern), and the sparse version ends up about as slow as dense.

**Why the crossover rises with more ranks.** Every added rank puts its nonzeros in new places, so the sum gets denser. At 99% sparsity and a vector length of 1 million, for example, the sum is about 8% nonzero with 8 ranks but about 72% with 128, and the sparse version goes from winning by 3.6× to losing. To keep the sum small enough for the sparse version to win, the inputs must be sparser, and the more ranks there are, the sparser they must be. Dense `MPI_Allreduce` also gets slower with more ranks (about 3 ms at 8 ranks, 17 ms at 128), which helps the sparse version a little, but not enough to make up for the fill-in.

### Fixing a slow step in the two-layer network

**What I saw.** With independent patterns and a vector length of 1 million per rank, the two-layer network was 2.3–3.2× slower than one layer.

**The clue.** The extra time grew with the size of the result. At 64 ranks and a vector length of 1 million, before the fix:

| Sparsity | Pairs in the result | Extra time over one layer | Extra time per result pair |
| --- | ---: | ---: | ---: |
| 90% | 998,801 | 27.2 ms | 27 ns |
| 95% | 962,565 | 27.2 ms | 28 ns |
| 98% | 726,270 | 20.9 ms | 29 ns |
| 99% | 475,651 | 12.6 ms | 26 ns |
| 99.5% | 274,840 | 10.0 ms | 36 ns |
| 99.9% | 61,884 | 2.3 ms | 37 ns |

From 90% to 99% sparsity, the result roughly halves and so does the extra time. So the extra cost came from work done once for every pair in the result, about 26–37 ns each. A one-off cost, like creating the second layer's communicators, wouldn't change with sparsity.

**The cause.** Only one step in the two-layer code touches every pair of the final result: a heap merge over the entire gathered result, done on every rank at `O(n log p)`. It wasn't needed. After two layers, each rank owns its own range of indices, so the gathered blocks never overlap; they just arrive in the wrong order.

**The fix.** Sort the blocks by their first index and join them, which takes linear time. Two-layer time with 64 ranks, a vector length of 1 million per rank, and independent patterns:

| Sparsity | 90% | 95% | 98% | 99% | 99.5% | 99.9% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 49.3 ms | 45.2 ms | 33.7 ms | 20.7 ms | 14.1 ms | 3.5 ms |
| After | 27.7 ms | 23.9 ms | 18.6 ms | 11.7 ms | 7.3 ms | 2.1 ms |
| One layer, for comparison | 21.4 ms | 18.4 ms | 13.6 ms | 8.2 ms | 4.5 ms | 1.2 ms |

**The outcome.** Two-layer time dropped by 39–49%, and the extra cost per result pair fell to 6–14 ns. In the layer comparison (96 configurations, vector length 1 million), two layers now beat one layer in 57 cases with a shared pattern, up from 32. With independent patterns, one layer is still faster in 95 of 96, likely because the second round of exchanges has to send partial sums that have already filled in.

## How it works

Each rank starts with a list of its nonzero (index, value) pairs sorted by index. At the end, every rank has the full sum, also sorted by index.

1. **Split into groups.** The size of each network layer (its "degree") decides which ranks talk to each other. The code supports one or two layers, and the degrees must multiply to the number of ranks.
2. **Send each pair to its owner.** The index range is divided among the ranks in a group. Each rank tells the others how many pairs it will send, then sends them with `MPI_Alltoallv`.
3. **Add up locally.** Each rank merges the sorted lists it received with a heap, adding pairs with the same index and dropping any that sum to zero.
4. **Share the result.** Ranks tell each other how many pairs they hold, then collect everyone's pairs with `MPI_Allgatherv`. With one layer, the pieces already arrive in index order. With two layers, each rank still owns a separate index range but the pieces arrive out of order, so they are sorted by first index and joined.

The network design comes from Huasha Zhao and John Canny's *Kylix: A Sparse Allreduce for Commodity Clusters* (ICPP 2014). This project implements a simplified version with one or two layers.

## How the measurements were taken

| Program | Warm-up runs | Timed runs | What is reported |
| --- | ---: | ---: | --- |
| `mpi` | 3 | 5 | For each timed run, the time of the slowest rank; the median of the five is reported |
| `mpi_runtime` | 3 | 1 | Time spent in each step on rank 0, plus the total, from one run |

Every run starts from the same input and checks the result on every rank, warm-ups included. Sparse and dense are timed separately with `MPI_Wtime`, each right after a barrier. The sparse time covers everything the operation does: creating and freeing communicators, communication, local merging, and assembling the final result. It leaves out generating the input, converting it to dense form, the barriers, collecting the timings, and the correctness check.

The profiler (`mpi_runtime`) splits the sparse time into communicator setup ("split"), scatter, merge ("reduce"), Allgather, and whatever is left ("other"). The notebook's plots combine split and other into one "remainder". These numbers come from a single run on rank 0, so they show where the time goes.

### Limitations

- Everything runs on **one node, with up to 128 ranks**. The results don't show how this behaves across several machines.
- Nonzero positions are random and spread evenly. With `-p shared`, every rank uses the same random seed to generate the input and gets **exactly the same positions and values**. With `-p independent`, each rank gets its own random positions. Real workloads usually sit somewhere in between, because common indices tend to show up on many ranks.
- All nonzero values are the integer 1. The sparsity setting is a target, so the actual number of nonzeros varies a little.
- Dense `MPI_Allreduce` is noticeably faster when the number of ranks is a power of two (for example, with a vector length of 1 million, 2.9 ms at 8 ranks vs 4.8 ms at 6). Speedups at 6, 12, 18, and 24 ranks are partly due to slow dense runs, so power-of-two rank counts give the fairest comparison.

## Build and run

### Requirements

- A C++17 compiler (the recorded results used GCC).
- MPI, with C++ compiler wrappers and CMake support.
- CMake 3.14 or newer.
- For the analysis only: Python with Jupyter, NumPy, pandas, and Matplotlib.

On Linux with MPI installed, from the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4

# Four ranks, one layer, shared pattern
mpiexec -n 4 ./build/mpi -n 1000000 -l 99 -p shared -s 12 -c 4

# Four ranks, two layers, independent patterns
mpiexec -n 4 ./build/mpi -n 1000000 -l 99.5 -p independent -s 12 -c 2 2
```

These are just examples; the recorded results used the configurations listed below. Use `mpi_runtime` instead of `mpi` to get the per-step breakdown. A successful run prints its timings followed by `Results Match!`. If the check fails, the program exits with a nonzero status.

| Option | What it does |
| --- | --- |
| `-n` | Vector length per rank, counting zeros and nonzeros (not the number of processes) |
| `-l` | Percentage of entries that are zero, from 0 to 100; decimals like `99.5` work |
| `-p` | `shared` (the default: every rank has the same nonzero positions) or `independent` (each rank has its own) |
| `-s` | Random seed; any nonzero value gives the same input every time, and 0 picks a random seed |
| `-c` | One or two layer degrees that multiply to the number of ranks (required) |
| `-h` | Show usage |

### Experiment configurations

Each log in `scaling/shared/` and `scaling/independent/` is one sweep over the settings below, run once with `-p shared` and once with `-p independent`. Every sweep covers 90, 95, 98, 99, 99.5, and 99.9% sparsity with seed 12 (`-s 12`).

| Log | Program | Ranks | Vector length per rank | Network (`-c`) |
| --- | --- | --- | --- | --- |
| `one_layer.txt` | `mpi` | 64 | 10K, 20K, 50K, 100K, 200K, 500K, 1M, 2M, 5M | One layer (`-c 64`) |
| `two_layer.txt` | `mpi` | 64 | Same as `one_layer.txt` | 16 × 4 (`-c 16 4`) |
| `procs_res.txt` | `mpi` | 2, 4, 6, 8, 12, 16, 18, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128 | 1M | One layer (`-c p`) |
| `layer_res.txt` | `mpi` | 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128 | 1M | One layer and two layers in the same job: (p/4) × 4, or (p/2) × 2 for p ≤ 16 |
| `runtime_res.txt` | `mpi_runtime` | Same as `procs_res.txt` | 1M | One layer (`-c p`) |

Each log is the program's output for the whole sweep. It starts with a `Pattern:` line, has a `Sparsity level:` line before each level, and a `Number of elements:` or `Number of Processes:` line before each run. In `layer_res.txt`, each run is also labeled `Single Layer` or `Two Layers`. The notebook reads these lines.

### Analysis notebook

To look at the recorded results without running anything:

```bash
python -m pip install jupyterlab numpy pandas matplotlib
cd scaling
jupyter lab analysis.ipynb
```

Run the notebook from `scaling/` so it can find the logs. Running all cells redraws the figures in `scaling/figures/` from the logs in `shared/` and `independent/`, including how dense the sum gets, and prints the crossover range for each rank count. Sections whose logs are missing are skipped.

## Repository layout

| Path | What's there |
| --- | --- |
| [`src/`](src/) | The sparse Allreduce library: data types, input generation, partitioning, merging, and the full `allreduce` call, with optional per-step timers |
| [`main.cpp`](main.cpp) | Benchmark program (`mpi`) |
| [`runtime/`](runtime/) | Profiling program (`mpi_runtime`), built on the same library |
| [`scaling/`](scaling/) | Analysis notebook, raw logs in `shared/` and `independent/`, and figures in `figures/` |
| [`report.pdf`](report.pdf) | Full report |

## License

MIT. See [LICENSE](LICENSE).
