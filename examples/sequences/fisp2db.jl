"""
    FISP2DB{T, Ns, U<:AbstractVector, V<:AbstractMatrix} <: EPGSimulator{T,Ns}

This struct is used to simulate gradient-spoiled sequence with varying flip angle
scheme. This struct is the same struct as used in fisp2d.jl, but as addition, the 
velocity of blood flow in axial direction is included. 

# Fields
- `RF_train::U` Vector with flip angle for each TR with abs.(RF_train) the RF flip angles in degrees and
    angle.(RF_train) should be the RF phases in degrees.
- `sliceprofiles::V` # Matrix with RF scaling factors (a.u.) to simulate slice profile effects.
    Each column represents the (flip angle dependent) scaling factors for one position along the slice direction.
- `TR::T`: Repetition time in seconds, assumed constant during the sequence
- `TE::T`: Echo time in seconds, assumed constant during the sequence
- `max_state::Val{Ns}`: Maximum number of states to keep track of in EPG simulation
- `TI::T`: Inversion delay after the inversion prepulse in seconds
- `v::T`:
"""

struct FISP2DB{T, Ns, U<:AbstractVector, V<:AbstractMatrix} <: EPGSimulator{T,Ns}
    RF_train::U
    sliceprofiles::V
    TR::T 
    TE::T
    max_state::Val{Ns}
    TI::T
    V::T
    H::T

    # TODO: maybe add inner construction which does some sanity checks
end

# To be able to change precision and send to CUDA device
@functor FISP2DB
@adapt_structure FISP2DB

# Methods needed to allocate an output array of the correct size and type
output_dimensions(sequence::FISP2DB) = length(sequence.RF_train)
output_eltype(sequence::FISP2DB) = unitless(eltype(sequence.RF_train))

# If RF doesn't have phase, configuration states will be real
Ω_eltype(sequence::FISP2DB) = unitless(eltype(sequence.RF_train))

# Sequence implementation
@inline function simulate_magnetization!(magnetization, sequence::FISP2DB, Ω, p::AbstractTissueParameters)

    T₁, T₂ = p.T₁, p.T₂
    TR, TE, TI, V, H = sequence.TR, sequence.TE, sequence.TI, sequence.V, sequence.H
    L = V * TR #Travel distance for blood on each TR



    E₁ᵀᴱ, E₂ᵀᴱ = E₁(Ω, TE, T₁),    E₂(Ω, TE, T₂)
    E₁ᵀᴿ⁻ᵀᴱ, E₂ᵀᴿ⁻ᵀᴱ = E₁(Ω, TR-TE, T₁), E₂(Ω, TR-TE, T₂)
    E₁ᵀᴵ, E₂ᵀᴵ = E₁(Ω, TI, T₁),    E₂(Ω, TI, T₂)

    eⁱᴮ⁰⁽ᵀᴱ⁾ = off_resonance_rotation(Ω, TE, p)
    eⁱᴮ⁰⁽ᵀᴿ⁻ᵀᴱ⁾ = off_resonance_rotation(Ω, TR-TE, p)
    z_body_track = 0.0 #keeps track of overall longitudinal magnetization due to inversion pulse
    @inbounds for spc in eachcol(sequence.sliceprofiles)

        initial_conditions!(Ω)
        # apply inversion pulse
        invert!(Ω)
        decay!(Ω, E₁ᵀᴵ, E₂ᵀᴵ)
        regrowth!(Ω, E₁ᵀᴵ)
        z_body_track = Z(Ω)[0,0]

        for (TR,RF) in enumerate(sequence.RF_train)

                # mix states
                excite!(Ω, spc[TR]*RF, p)
        #      # T2 decay F states, T1 decay Z states, B0 rotation until TE
                rotate_decay!(Ω, E₁ᵀᴱ, E₂ᵀᴱ, eⁱᴮ⁰⁽ᵀᴱ⁾)
        #         regrowth!(Ω, E₁ᵀᴱ)            # sample F₊[0]
                z_body_track += (1-E₁ᵀᴱ)
                 sample_transverse_V2!(magnetization, TR, Ω)
        #         # T2 decay F states, T1 decay Z states, B0 rotation until next RF excitation
                rotate_decay!(Ω, E₁ᵀᴿ⁻ᵀᴱ, E₂ᵀᴿ⁻ᵀᴱ, eⁱᴮ⁰⁽ᵀᴿ⁻ᵀᴱ⁾)
                regrowth!(Ω, E₁ᵀᴿ⁻ᵀᴱ)
                z_body_track += (1-E₁ᵀᴿ⁻ᵀᴱ)
        #        # shift F states due to dephasing gradients
                dephasing!(Ω)
                blood_shift!(Ω, z_body_track)
        end
    end
    return nothing
end

# Add method to getindex to reduce sequence length with convenient syntax (idx is something like 1:nr_of_readouts)
Base.getindex(seq::FISP2DB, idx) = typeof(seq)(seq.RF_train[idx], seq.sliceprofiles, seq.TR, seq.TE, seq.max_state, seq.TI)

# The _value_ of max_state needs to be part of the type, not its type (<:Int)
# That's what the Val{Ns} thing does. Because it's easy to forget doing Val(max_state) when constructing FISP2DB,
# here's a constructor that takes care of it in case you forget.
FISP2DB(RF_train, sliceprofiles, TR, TE, max_state::Int, TI, V, H) = FISP2DB(RF_train, sliceprofiles, TR, TE, Val(max_state), TI, V, H)

# Nicer printing of sequence in REPL
# Base.show(io::IO, ::MIME"text/plain", seq::FISP2DB) = begin
Base.show(io::IO, seq::FISP2DB) = begin
    println("")
    println(io, "FISP2DB sequence")
    println(io, "RF_train:     ", typeof(seq.RF_train), " $(length(seq.RF_train)) flip angles")
    println(io, "sliceprofiles:", "$(typeof(seq.sliceprofiles)) $(size(seq.sliceprofiles))")
    println(io, "TR:           ", seq.TR)
    println(io, "TE:           ", seq.TE)
    println(io, "max_state:    ", seq.max_state)
    println(io, "TI:           ", seq.TI)
end

# Convenience constructor to quickly generate pSSFP sequence of length nTR
FISP2DB(nTR) = FISP2DB(complex.(ones(nTR)), complex.(ones(nTR,3)), 0.010, 0.005, Val(5), 0.1)

# Constructor for sequence without slice profile correction
FISP2DB(RF_train, TR, TE, max_state, TI, V, H) = begin
    sliceprofiles = ones(eltype(RF_train), length(RF_train), 1)
    FISP2DB(RF_train, sliceprofiles, TR, TE, max_state, TI, V, H)
end

export FISP2DB