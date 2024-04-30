# # MR Fingerprinting Dictionary Generation

#using Pkg; Pkg.activate("docs")

# In this example we demonstrate how to generate an MR Fingerprinting 
# dictionary using a FISP type sequence

using BlochSimulators
using ComputationalResources

nTR = 1000; # nr of TRs used in the simulation
RF_train = LinRange(1,90,nTR) |> collect; # flip angle train
TR,TE,TI= 0.010, 0.005, 0.100; # repetition time, echo time, inversion delay, blood velocity
max_state = 25; # maximum number of configuration states to keep track of 
H = 0.004; #Slice thickness in ??
V = 0.0004; # Blood velocity in m/s

sequence = FISP2DB(RF_train, TR, TE, max_state, TI, V, H);

T₁ = 1.5; # T₁ range 
T₂ = 0.24; # T₂ range

parameters = map(T₁T₂, Iterators.product(T₁,T₂)); # produce all parameter pairs
parameters = filter(p -> (p.T₁ > p.T₂), parameters); # remove pairs with T₂ ≤ T₁

println("Length parameters: $(length(parameters))")

# Now we can perform the simulations using different hardware resources
# 
# Note that the first time a function is called in a Julia session,
# a precompilation procedure starts and the runtime for subsequent function
# calls are significantly faster

@time dictionary = simulate_magnetization(CPU1(), sequence, parameters);

using Plots 
x = 1:nTR;
y = dictionary[:,1];


plot1 = plot(x,y, xlabel="TR", ylabel="F+", title="Magnetization evolution");
plot2 = plot(x, RF_train, xlabel="TR", ylabel="Flip angle", title="Flip angle train");
plot(plot1, plot2, layout = (2,1), legend=false, size=(800,600))
