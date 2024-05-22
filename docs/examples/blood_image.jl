# # MR Fingerprinting Dictionary Generation

#using Pkg; Pkg.activate("docs")

# In this example we demonstrate how to generate an MR Fingerprinting 
# dictionary using a FISP type sequence

using Revise
using BlochSimulators
using ComputationalResources
using MAT
using Plots

#Initalizing conditions for blood
TR,TE,TI= 0.0089, 0.005, 0.100;
max_state = 10;
nTR = 1120; # nr of TRs used in the simulation
T₁ = 1.584 #exact T1 value for blood
T₂ = 0.165 #exact T2 value for blood
H = 0.004*3;
parameters = map(T₁T₂, Iterators.product(T₁,T₂)); # produce all parameter pairs
parameters = filter(p -> (p.T₁ > p.T₂), parameters); # remove pairs with T₂ ≤ T₁

file = matopen("docs/examples/FA_insync.mat")
RF_train = read(file, "fa") |> vec 
close(file) 

#Simulating without velocity (Vb = 0)
old_method = FISP2D(RF_train, TR, TE, max_state, TI)

cu_sequence = old_method |> f32 |> gpu;
cu_parameters = parameters |> f32 |> gpu;

# Remember, the first time a compilation procedure takes place which, especially
# on GPU, can take some time.
println("Active CUDA device:"); BlochSimulators.CUDA.device()

@time simulation_old = simulate_magnetization(CUDALibs(), cu_sequence, cu_parameters);

simulation_range = LinRange(0.5, 0.01, 5)
simulation_new = zeros(nTR, length(simulation_range))

for (i , value) in enumerate(simulation_range)
    Vb = value
    sequence = FISP2DB(RF_train, TR, TE, max_state, TI, Vb, H);
    cu_sequence = sequence |> f32 |> gpu;
    @time simulation = simulate_magnetization(CUDALibs(), cu_sequence, cu_parameters);
    simulation_new[:, i] = simulation[:,1]
end
p = plot()
for (i, value) in enumerate(simulation_range)
    value = round(value, digits = 3)
    plot!(p, simulation_new[:,i], label = "Vᵦ = $value", color = "red")
    annotate!(p, 150, simulation_new[350,i], text("Vᵦ = $value", 8, :left, color = "red"))
end
plot!(p, simulation_old[:,1], label = "Vᵦ = 0", color = "Green", title = "MR signal simulations from gradient spoiled \n sequence using the new implementation \n including velocities (red) and the old implementation \n without velocities (green) \n", xlabel = "TR", ylabel = "Magnetization signal", legend = :topright, size = (800, 600))

rftrain = plot()
plot!(rftrain, RF_train, title = "RF train", xlabel = "TR", ylabel = "Flip angle", size = (800, 600), legend=false)