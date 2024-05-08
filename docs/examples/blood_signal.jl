# # MR Fingerprinting Dictionary Generation

#using Pkg; Pkg.activate("docs")

# In this example we demonstrate how to generate an MR Fingerprinting 
# dictionary using a FISP type sequence

using Revise
using BlochSimulators
using ComputationalResources
using MAT
using Plots


nTR = 1120; # nr of TRs used in the simulation
RF_train = LinRange(1,90,nTR) |> collect; # flip angle train
# file = matopen("docs/examples/FA_insync.mat")
# RF_train = read(file, "fa") |> vec 
# close(file) 


TR,TE,TI= 0.0089, 0.005, 0.100; # repetition time, echo time, inversion delay, blood velocity
max_state = 10; # maximum number of configuration states to keep track of 
H = 0.004*3; #Slice thickness in m
# V = 0.328; # Blood velocity in m/s
Vb = 0.32;
#sequence = FISP2DB(RF_train, TR, TE, max_state, TI, Vb, H);
 sequence = FISP2D(RF_train, TR, TE, max_state, TI)

# T₁ = LinRange(1.584-0.005, 1.584+0.005, 10) |> collect; # T₁ range 
# T₂ = LinRange(0.165-0.0032, 0.165+0.0032, 10) |> collect; # T₂ range
T₁ = 1.584
T₂ = 0.165

parameters = map(T₁T₂, Iterators.product(T₁,T₂)); # produce all parameter pairs
parameters = filter(p -> (p.T₁ > p.T₂), parameters); # remove pairs with T₂ ≤ T₁

println("Length parameters: $(length(parameters))")

# Now we can perform the simulations using different hardware resources
# 
# Note that the first time a function is called in a Julia session,
# a precompilation procedure starts and the runtime for subsequent function
# calls are significantly faster

println("Current number of threads: $(Threads.nthreads())")
@time dictionary = simulate_magnetization(CPUThreads(), sequence, parameters);

 
x = 1:nTR;
x_matrix = repeat(x, 1, length(parameters))
y = dictionary;


super_title = "FISP2DB_With_Blood_Velocity_NewMethod"
title_plot = title = plot(title = super_title, grid = false, showaxis = false, bottom_margin = -50Plots.px);
plot1 = plot(x_matrix,y, xlabel="TR", ylabel="F+", title="Magnetization evolution");
plot2 = plot(x, RF_train, xlabel="TR", ylabel="Flip angle", title="Flip angle train");
plot_combi = plot(title_plot, plot1, plot2, layout =  @layout([A{0.01h}; [B C]]), legend=false, size=(800, 600))


savefig(plot_combi, string("C:/Users/20212059/OneDrive - TU Eindhoven/Documents/School/BEP/Data/", super_title, ".png"))