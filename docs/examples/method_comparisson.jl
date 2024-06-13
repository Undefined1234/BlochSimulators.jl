# Voxel partitioning tests
# This example was made to test how the new FISP2DB performes on voxels containing blood velocities 
# A simulation is performed using FISP2DB with correct T1 and T2 for blood
# Resulting simulation if then fitted using a dictionary containing simulations from the old FISP2D
# Resulting fit should show a significantly higher T1 and T2 than the correct values

using Revise
using BlochSimulators
using ComputationalResources
using MAT
using Plots
using Statistics
using StatsBase
using FastGaussQuadrature

nTR = 1120; # nr of TRs used in the simulation
# RF_train = LinRange(1,90,nTR) |> collect; # flip angle train
file = matopen("docs/examples/FA_insync.mat")
RF_train = read(file, "fa") |> vec 
close(file) 

TR,TE,TI= 0.0089, 0.005, 0.100; # repetition time, echo time, inversion delay, blood velocity
max_state = 10; # maximum number of configuration states to keep track of 
H = 0.004*3; #Slice thickness in m
# V = 0.328; # Blood velocity in m/s
Vb = 0.32;

sym_FISP2DB = FISP2DB(RF_train, TR, TE, max_state, TI, Vb, H); # FISP2DB sequence for blood
sym_FISP2D = FISP2D(RF_train, TR, TE, max_state, TI); # FISP2D sequence for dictionary

T₁ = 2.0855 #exact T1 value for blood
T₂ = 0.275 #exact T2 value for blood

parameters = map(T₁T₂, Iterators.product(T₁,T₂)); # produce all parameter pairs
parameters = filter(p -> (p.T₁ > p.T₂), parameters); # remove pairs with T₂ ≤ T₁

println("Length parameters: $(length(parameters))")

cu_sequence_FISP2D = sym_FISP2D |> f32 |> gpu;
cu_sequence_FISP2DB = sym_FISP2DB |> f32 |> gpu;
cu_parameters = parameters |> f32 |> gpu;

# Remember, the first time a compilation procedure takes place which, especially
# on GPU, can take some time.
println("Active CUDA device:"); BlochSimulators.CUDA.device()

@time FISP2D_path = simulate_magnetization(CUDALibs(), cu_sequence_FISP2D, cu_parameters);
@time FISP2DB_path = simulate_magnetization(CUDALibs(), cu_sequence_FISP2DB, cu_parameters);
# Call the pre-compiled version


#Evaluation 
d = FISP2D_path .- FISP2DB_path
d_sum_abs = sum(abs.(d))

p = plot()
plot!(p, FISP2D_path, label = "FISP2D", color = "red")
plot!(p, FISP2DB_path, label = "FISP2DB", color = "blue")
plot!(p, d, label = "Difference", color = "green", title = "MR signal simulations for simulation 3", ylabel = "Magnetizaion signal", xlabel = "TR")


# savefig(plot1, "C:/Users/20212059/OneDrive - TU Eindhoven/Documents/School/BEP/GPU.png")