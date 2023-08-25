# Simulate MR Signal

In this example we are going to generate a phantom
and simulate the signal for a gradient-balanced sequence
with a Cartesian gradient trajectory

````julia
using Revise
using BlochSimulators
using ComputationalResources
using StaticArrays
using LinearAlgebra
using PyPlot
using FFTW
import ImagePhantoms
````

First we assemble a Shepp Logan phantom with homogeneous T₁ and T₂
but non-constant proton density and B₀

````julia
N = 256
ρ = complex.(ImagePhantoms.shepp_logan(N, ImagePhantoms.SheppLoganEmis())');
T₁ = fill(0.85, N, N);
T₂ = fill(0.05, N, N);
B₀ = repeat(1:N,1,N);
````

We also set the spatial coordinates for the phantom

````julia
FOVˣ, FOVʸ = 25.6, 25.6;
X = [x for x ∈ LinRange(-FOVˣ/2, FOVˣ/2, N), y ∈ 1:N];
Y = [y for x ∈ 1:N, y ∈ LinRange(-FOVʸ/2, FOVʸ/2, N)];
````

Finally we assemble the phantom as an array of `T₁T₂B₀ρˣρʸxy` values

````julia
parameters = map(T₁T₂B₀ρˣρʸxy, T₁, T₂, B₀, real.(ρ), imag.(ρ), X, Y);
````

Next, we assemble a balanced sequence with constant flip angle of 60 degrees,
a TR of 10 ms and 0-π phase cycling. See `src/sequences/pssfp.jl` for the
sequence description and the fields for which values must be provided

````julia
nTR = N
RF_train = complex.(fill(40.0, nTR)) # constant flip angle train
RF_train[2:2:end] .*= -1 # 0-π phase cycling
nRF = 25 # nr of RF discretization points
durRF = 0.001 # duration of RF excitation
TR = 0.010 # repetition time
TI = 10.0 # long inversion delay -> no inversion
gaussian = [exp(-(i-(nRF/2))^2 * inv(nRF)) for i ∈ 1:nRF] # RF excitation waveform
γΔtRF = (π/180) * normalize(gaussian, 1) |> SVector{nRF} # normalize to flip angle of 1 degree
Δt = (ex=durRF/nRF, inv = TI, pr = (TR - durRF)/2); # time intervals during TR
γΔtGRz = (ex=0.002/nRF, inv = 0.00, pr = -0.01); # slice select gradient strengths during TR
nz = 35 # nr of spins in z direction
z = SVector{nz}(LinRange(-1,1,nz)) # z locations

sequence = pSSFP(RF_train, TR, γΔtRF, Δt, γΔtGRz, z)
````

Next, we assemble a Cartesian trajectory with linear phase encoding
(see `src/trajectories/cartesian.jl`).

````julia
nr = N # nr of readouts
ns = N # nr of samples per readout
Δt_adc = 10^-5 # time between sample points
py = -(N÷2):1:(N÷2)-1 # phase encoding indices
Δkˣ = 2π / FOVˣ; # k-space step in x direction for Nyquist sampling
Δkʸ = 2π / FOVʸ; # k-space step in y direction for Nyquist sampling
k0 = [(-ns/2 * Δkˣ) + im * (py[r] * Δkʸ) for r in 1:nr]; # starting points in k-space per readout
Δk = [Δkˣ + 0.0im for r in 1:nr]; # k-space steps per sample point for each readout

trajectory = CartesianTrajectory(nr,ns,Δt_adc,k0,Δk,py);
````

We use two different receive coils

````julia
coil₁ = complex.(repeat(LinRange(0.5,1.0,N),1,N));
coil₂ = coil₁';

coil_sensitivities = map(SVector, coil₁, coil₂)
````

Now simulate the signal for the sequence, trajectory, phantom and coils on GPU

````julia
resource = CUDALibs()

sequence            = gpu(f32(sequence))
parameters          = gpu(f32(parameters))
trajectory          = gpu(f32(trajectory))
coil_sensitivities  = gpu(f32(coil_sensitivities))

signal = simulate(CUDALibs(), sequence, parameters, trajectory, coil_sensitivities)

signal = collect(signal)
````

Let's look at fft images of the signals from the two different coils

````julia
signal₁ = first.(signal)
signal₂ = last.(signal)


@. signal₁[2:2:end] *= -1 # correct for phase cycling
@. signal₂[2:2:end] *= -1 # correct for phase cycling

fft_image₁ = rot180(ifft(reshape(signal₁,N,N))) # rot180 instead of fftshifts
fft_image₂ = rot180(ifft(reshape(signal₂,N,N))) # rot180 instead of fftshifts

figure()
subplot(1,3,1); imshow(abs.(ρ)); title("Ground truth \n proton density")
subplot(1,3,2); imshow(abs.(fft_image₁)); title("fft-image coil 1")
subplot(1,3,3); imshow(abs.(fft_image₂)); title("fft-image coil 2")
````

Note the banding due to off-resonance

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
