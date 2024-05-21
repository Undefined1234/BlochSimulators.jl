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
sequence_blood = FISP2DB(RF_train, TR, TE, max_state, TI, Vb, H); # FISP2DB sequence for blood

T₁ = 1.584 #exact T1 value for blood
T₂ = 0.165 #exact T2 value for blood

parameters_blood = map(T₁T₂, Iterators.product(T₁,T₂)); # produce all parameter pairs
parameters_blood = filter(p -> (p.T₁ > p.T₂), parameters_blood); # remove pairs with T₂ ≤ T₁

println("Length parameters: $(length(parameters_blood))")

println("Current number of threads: $(Threads.nthreads())")
@time blood_sim = simulate_magnetization(CPUThreads(), sequence_blood, parameters_blood);

#Creating dictionary with old FISP2D sequence
#
T₁ = .1:.1:3 #T1 range for old simulation
T₂ = .01:.01:1 #T2 range for old simulation

sequence = FISP2D(RF_train, TR, TE, max_state, TI);

parameters = map(T₁T₂, Iterators.product(T₁,T₂)); # produce all parameter pairs
parameters = filter(p -> (p.T₁ > p.T₂), parameters); # remove pairs with T₂ ≤ T₁

println("Length parameters: $(length(parameters))")

println("Current number of threads: $(Threads.nthreads())")
@time dictionary = simulate_magnetization(CPUThreads(), sequence, parameters);


#Evaluation 
#
evaluation = zeros(size(dictionary)[2]);
for i in 1:size(dictionary)[2]
    evaluation[i] = L2dist(blood_sim, dictionary[:,i]);
end
val, index = findmin(evaluation)
T1 = parameters[index[1]][1]*1000
T2 = parameters[index[1]][2]*1000

x = 1:nTR;
y_blood = blood_sim;
y_sim = dictionary[:,index[1]];
plot1 = plot(x, [y_blood y_sim], labels=["FISP2DB" "FISP2D"], ylabel="Magnetization", xlabel="TR")

println("Max correlation: $val with T1: $T1 ms and T2: $T2 ms")
