```@meta
CurrentModule = BlochSimulators
```

# Overview

## MR Signal Equation

In Magnetic Resonance Imaging (MRI), the complex signal ``s`` at time point ``t`` of a pulse sequence can be modelled as

``s(t) = \int_V c(\vec{r})m_{\perp}(\vec{r},t)d\vec{r}``. 

Here ``c`` is the spatially-dependent receive sensitivity of the coil used for signal reception, ``V`` is field-of-view of the receive coil and ``m_{\perp}(\vec{r},t)`` is the (complex) transverse magnetization at the spatial location ``\vec{r}`` at time point ``t``. 

The dynamical behaviour of the magnetization is described by the [Bloch equations](https://en.wikipedia.org/wiki/Bloch_equations) and depends on the pulse sequence used in the acquisition, tissue parameters (e.g. ``T_1``, ``T_2`` and proton density) and system parameters (e.g. ``\Delta B_0`` and `B_{1}^{+}` inhomogeneities). Given a pulse sequence as well as tissue- and system parameters in a voxel with spatial location ``\vec{r}``, the transverse magnetization ``m_{\perp}(r,t)`` at arbitrary time points ``t`` can be obtained by numerical integration of the Bloch equations (i.e. _Bloch simulations_). Note that ``m_{\perp}(\vec{r},t)`` can in fact be separated as 

``m_{\perp}(\vec{r},t) = m(\vec{r},t)e^{-2\pi i \vec{k}(t)\cdot \vec{r}}``, 

where ``m`` is the transverse magnetization without any (in-plane / in-slab) gradient encoding, and ``\vec{k}(t)`` is the gradient trajectory used in the acquisition. That is, ``m`` and the gradient encoding ``e^{-2\pi i \vec{k}(t)\cdot \vec{r}}`` can be computed separately. 

To simulate ``s`` in practice, spatial discretization must first be performed. Divide the field-of-view into ``N_{v}`` voxels spatial coordinates ``\vec{r}_{1}, \ldots, \vec{r}_{N_v}``. Assume each voxel to have the same volume ``\Delta V``. The signal ``s(t)`` is then computed as the discrete sum

``s(t) = \sum_{j=1}^{N_v} c(\vec{r}_j)m(\vec{r}_j,t)e^{-2\pi i \vec{k}(t)\cdot \vec{r}_j} \Delta V``.

BlochSimulators provides tools to perform Bloch simulations (i.e. numerical integration of the Bloch equations) for computing ``m`` and to evaluate the discrete sum required for computing the signal ``s`` at desired sample times. 

## Design philosophy

Whereas other Bloch simulation toolboxes typically provide a single, generic simulator that can be used for arbitrary pulse sequences, with BlochSimulators one is encouraged to assemble pulse sequence-specific simulators. The philosophy behind this design choice is that sequence-specific simulators can incorporate knowledge of repetitive patterns in pulse sequence (e.g. RF excitation waveform and/or gradient waveforms) to improve the runtime performance of the simulator. 

A second important design philosophy is that, when one intends to simulate MR signal for some numerical phantom, first the magnetization ``m`` at echo times in all voxels is computed (without taking into account the effects of the gradient trajectory). If ``m_e`` is the transverse magnetization in one voxel at some echo time of the pulse sequence, then the transverse magnetization at the ``j``-th sample point of that readout relative to the echo time can be computed analytically as 

``m_s = m_e \left( e^{-2\pi i \Delta B_0 \Delta t}e^{-\Delta t / T_2})^j``,

where ``\Delta t`` is the time between sample points of the readout.

Given ``m`` at echo times, the signal ``s`` is computed by _expanding_ the magnetization to all sample times and evaluating the discrete sum while taking into account the gradient trajectory (and coil sensitivities). 

Note that the (discretized) signal equation closely resembles a Discrete Fourier Transform. In many MRI applications, the Fast Fourier Transform (FFT) is used to transform back-and-forth between the image domain and the k-space domain. In BlochSimulators, we intentionally do not rely on the FFT since it does not allow to take into account dynamical behaviour during readouts (e.g. ``T_2`` decay and/or ``\Delta B_0``-induced rotations).

## Simulating magnetization at echo times

BlochSimulators supports two different models for performing Bloch simulations: the individual isochromat model and the extended phase graph model. For both models, basic operator functions are implemented (see [`src/operators/isochromat.jl`](https://github.com/oscarvanderheide/BlochSimulators.jl/blob/main/src/operators/isochromat.jl) and [`src/operators/epg.jl`](https://github.com/oscarvanderheide/BlochSimulators.jl/blob/main/src/operators/epg.jl)) in a type-stable and non-allocating fashion. By combining these operators, simulators for entire pulse sequences can be assembled. To this end, a user must define a new `Struct` that is a subtype of either `IsochromatSimulator` or `EPGSimulator` with fields that are necessary to describe the pulse sequence (e.g. flip angle(s), TR, TE, etc.). A method must then be added for this new type to the `simulate!` function which, by combining the fields of the struct with the basic operators, implements the magnetization response of the sequence. See [`src/sequences/_interface.jl`](https://github.com/oscarvanderheide/BlochSimulators.jl/blob/main/src/sequences/_interface.jl) for additional information and requirements. 

To perform simulations, tissue parameter inputs must be provided. Custom structs for different combinations of tissue parameters are introduced in this package (all of which are subtypes of `AbstractTissueParameters`). See [`src/parameters/tissueparameters.jl`](https://github.com/oscarvanderheide/BlochSimulators.jl/blob/main/src/parameters/tissueparameters.jl) for more information. 

Given a `sequence` struct together with a set of input parameters  for each voxel (currently the parameters must be an `::AbstractArray{<:AbstractTissueParameters}`), the magnetization at echo times in each voxel is obtained with the function call 

`echos = simulate(resource, sequence, parameters)`,

where `resource` is either `CPU1()`, `CPUThreads()`, `CPUProcesses()` or `CUDALibs()` (see [ComputationalResources.jl](https://github.com/timholy/ComputationalResources.jl)). This function can be used in the context of MR Fingerprinting to generate a dictionary.

## Simulating the MR signal

To compute the MR signal ``s`` given the magnetization at echo times in all voxels, information from the gradient trajectory (i.e. ``k(t)``) is required. For each different type of trajectory, a new struct (subtype of `AbstractTrajectory`) must be introduced with fields that describe that particular gradient trajectory. A new method must then be added to `echos_to_signal` which evaluates 

``\sum_{j=1}^{N_v} c(\vec{r}_j)m(\vec{r}_j,t)e^{-2\pi i \vec{k}(t)\cdot \vec{r}_j} \Delta V``.

That is, the magnetization at echo times must be expanded to all sample points, the gradient trajectory and coil sensitivities must be taken into account and the summation over all voxels must be performed. 

See [`src/trajectories/_interface.jl`](https://github.com/oscarvanderheide/BlochSimulators.jl/blob/main/src/trajectories/_interface.jl) for more details. Example implementations for [Cartesian](https://github.com/oscarvanderheide/BlochSimulators.jl/blob/main/src/trajectories/cartsian.jl) and [radial](https://github.com/oscarvanderheide/BlochSimulators.jl/blob/main/src/trajectories/radial.jl) trajectories are provided.

Given the magnetization at echo times in all voxels (stored in ``echos``), the signal ``s`` at sample times is computed with the function call

`signal = echos_to_signal(resource, echos, parameters, trajectory, coil_sensitivities)`.

Alternatively, the signal can be computed with the `simulate` function as

`signal = simulate(resource, sequence, parameters, trajectory, coil_sensitivities)`.

## GPU Compatibility 

The operator functions for both the isochromat model and the extended phase graph model have been designed to be type-stable and non-allocating. A proper combination of these operators should make it possible to have type-stable and non-allocating simulations for entire pulse sequences. Besides being beneficial for performance on CPU hardware, this also makes it possible to run simulations on GPU cards (NVIDIA) using the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). 

To run simulations on a GPU card, the `sequence`, `trajectory`, `parameters` and `coil_sensitivities` must be sent to the GPU first. For this purpose, a convenience function `gpu` is exported. 

Also note that by default, Julia arrays have double precision (`Float64` or `ComplexF64`) upon construction. To convert to single precision (`Float32`), a convenience function `f32` is exported as well. 

For example, given some `sequence`, `gpu(f32(sequence))` will recursively convert its fields to single precision and convert any regular `Arrays` to `CuArrays`.

## Todo

- For 3D applications, storing the magnetization at echo times for all voxels may not be feasible. The computations can be performed in batches though but is currently not implemented.
- Add diffusion operators to both the isochromat and extended phase graph models.
- Add magnetization transfer model.
- Add spiral and EPI trajectories.
- Store `parameters` as `StructArray` rather than `AbstractArray{<:AbstractTissueParameters}`. Perhaps separate the spatial coordinates from the tissue parameters.