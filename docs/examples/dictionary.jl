# # MR Fingerprinting Dictionary Generation

# In this example we demonstrate how to generate an MR Fingerprinting 
# dictionary using a FISP type sequence

using BlochSimulators
using ComputationalResources

# First, construct a FISP sequence struct (see `src/sequences/fisp.jl` 
# for which fields are necessary and which constructors exist)

nTR = 1000; # nr of TRs used in the simulation
RF_train = LinRange(1,90,nTR) |> collect; # flip angle train
TR,TE,TI = 0.010, 0.005, 0.100; # repetition time, echo time, inversion delay
max_state = 25; # maximum number of configuration states to keep track of 

sequence = FISP(RF_train, TR, TE, max_state, TI);

# Next, set the desired input parameters

T₁ = 0.500:0.010:5.0; # T₁ range 
T₂ = 0.025:0.005:1.0; # T₂ range

parameters = map(T₁T₂, Iterators.product(T₁,T₂)); # produce all parameter pairs
parameters = filter(p -> (p.T₁ > p.T₂), parameters); # remove pairs with T₂ ≤ T₁

# Now we can perform the simulations using different hardware resources
# 
# Note that the first time a function is called in a Julia session,
# a precompilation procedure starts and the runtime for subsequent function
# calls are significantly faster
# 
# First, we simply simulate a dictionary using single-threaded CPU mode:

@time dictionary = simulate_echos(CPU1(), sequence, parameters);
@time dictionary = simulate_echos(CPU1(), sequence, parameters);

# To use multiple threads, Julia must be started with the `--threads=auto`
# flag (or some integer instead of `auto`). Then, we can simulate in a 
# multi-threaded fashion with the following syntax:

@time dictionary = simulate_echos(CPUThreads(), sequence, parameters);
@time dictionary = simulate_echos(CPUThreads(), sequence, parameters);

# For distributed CPU mode, use the Distribute packages (ships with Julia)
# to add workers first

using Distributed
addprocs(4, exeflags="--project=.")

# Alternatively, if you can ssh into some other machine, 
# you can add CPUs from that machine as follows:
addprocs([("12.345.67.89", 4)], exeflags="--project=.")

# Or, if you want to run this code on cluster with a queuing system, use ClusterManagers package.
# 
# After workers have been added, load BlochSimulators on all workers 
# and then start a distributed dictionary generation with:
@everywhere using BlochSimulators

@time dictionary = simulate_echos(CPUProcesses(), sequence, parameters);
@time dictionary = simulate_echos(CPUProcesses(), sequence, parameters);

# To perform simulations on GPU, we first convert the sequence and parameters
# to single precision and then send them to the gpu

cu_sequence = sequence |> f32 |> gpu;
cu_parameters = parameters |> f32 |> gpu;

@time dictionary = simulate_echos(CUDALibs(), cu_sequence, cu_parameters);
@time dictionary = simulate_echos(CUDALibs(), cu_sequence, cu_parameters);


