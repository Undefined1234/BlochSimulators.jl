module BlochSimulators

    # Load external packages
    using ComputationalResources
    using CUDA
    using Distributed
    using DistributedArrays
    using Functors
    using InteractiveUtils # needed for subtypes function
    using LinearAlgebra
    using OffsetArrays
    using StaticArrays
    using Unitful
    using Unitless

    import Adapt: adapt, adapt_storage, @adapt_structure # to allow custom Structs of non-isbits type to be used on gpu
    import Functors: @functor, functor, fmap, isleaf

    # hard-coded nr of threads per block on GPU
    const THREADS_PER_BLOCK = 32

    # To perform simulations we need tissue parameters as inputs.
    # Supported combinations of tissue parameters are defined
    # in tissueparameters.jl
    include("interfaces/tissueparameters.jl")

    export AbstractTissueParameters, hasB₁, hasB₀

    # Informal interface for sequence implementations. By convention,
    # a sequence::BlochSimulator is used to simulate magnetization at
    # echo times only.
    include("interfaces/sequences.jl")

    export BlochSimulator, IsochromatSimulator, EPGSimulator

    # Operator functions for isochromat model and EPG model
    include("operators/isochromat.jl")
    include("operators/epg.jl")
    include("operators/utils.jl")

    export Isochromat, EPGStates

    # Currently included example sequences:

    # Isochromat simulator that is generic in the sense that it accepts
    # arrays of RF and gradient waveforms similar to the Bloch simulator from
    # Brian Hargreaves (http://mrsrl.stanford.edu/~brian/blochsim/)
    include("../examples/sequences/generic2d.jl") # with summation over slice direction
    include("../examples/sequences/generic3d.jl")

    # An isochromat-based pSSFP sequence with variable flip angle train
    include("../examples/sequences/pssfp2d.jl")
    include("../examples/sequences/pssfp3d.jl")

    # An EPG-based gradient-spoiled (FISP) sequence with variable flip angle train
    include("../examples/sequences/fisp2d.jl")
    include("../examples/sequences/fisp3d.jl")
    include("../examples/sequences/fisp2db.jl")
    include("../examples/sequences/adiabatic.jl")

    # Informal interface for trajectory implementations. By convention,
    # a sequence::BlochSimulator is used to simulate magnetization at echo times
    # only and a trajectory::AbstractTrajectory is used to simulate
    # the magnetization at other readout times.
    include("interfaces/trajectories.jl")

    # Currently included example trajectories:
    include("../examples/trajectories/cartesian.jl")
    include("../examples/trajectories/radial.jl")

    # Utility functions (gpu, f32, f64) to send structs to gpu
    # and change their precision
    include("utils/gpu.jl")
    include("utils/precision.jl")

    # # This packages supports different types of computation:
    # # 1. Single CPU computation
    # # 2. Multi-threaded CPU computation (when Julia is started with -t <nthreads>)
    # # 2. Multi-process CPU computation (when workers are added with addprocs)
    # # 3. CUDA GPU computation
    # # ComputationalResources is used to dispatch on the different computational resources.

    # # Main function to call a sequence simulator with a set of input parameters are defined in simulate.jl
    include("simulate/magnetization.jl")
    include("simulate/signal.jl")

    export simulate_magnetization, simulate_signal, magnetization_to_signal, phase_encoding!

end # module
