"""
    EPGStates = Union{MMatrix{3}, SizedMatrix{3}}

In the EPG model, the configuration state matrix `Ω` will be updated inplace.
On CPU, we use `StaticArrays.MMatrix`. The MMatrix will not escape `_simulate!`
and therefore should not result in allocations.
On GPU, we use shared memory (`CUDA.CuStaticSharedArray`). The shared memory is allocated
for all threads within a block simultaneously. We then take a `@view` and wrap it in a `SizedArray`.
Without the `SizedArray`, the type would be long and unreadable. By wrapping, we can then
simply dispatch on a `SizedArray` instead.
"""
#const EPGStates = Union{MArray{Tuple{3, 10, 4}}, SizedArray{Tuple{3, 10, 4}}}
const EPGStates = Any


"""
    F₊(Ω)

View into the first row of the configuration state matrix `Ω`,
corresponding to the `F₊` states.
"""
F₊(Ω) = OffsetMatrix(view(Ω,1,:,:), 0:size(Ω,2)-1, 0:size(Ω,3)-1)


"""
    F̄₋(Ω)

View into the second row of the configuration state matrix `Ω`,
corresponding to the `F̄₋` states.
"""
F̄₋(Ω) = OffsetMatrix(view(Ω,2,:,:), 0:size(Ω,2)-1, 0:size(Ω,3)-1)
"""
    Z(Ω)

View into the third row of the configuration state matrix `Ω`,
corresponding to the `Z` states.
"""
Z(Ω) = OffsetMatrix(view(Ω,3,:,:), 0:size(Ω,2)-1, 0:size(Ω,3)-1)

## KERNELS ###

# Initialize States

"""
    Ω_eltype(sequence::EPGSimulator{T,Ns}) where {T,Ns} = Complex{T}

By default, configuration states are complex. For some sequences, they
will only ever be real (no RF phase, no complex slice profile correction)
and for these sequences a method needs to be added to this function.

"""
@inline Ω_eltype(sequence::EPGSimulator{T,Ns}) where {T,Ns} = Complex{T} 


"""
    initialize_states(::AbstractResource, sequence::EPGSimulator{T,Ns}) where {T,Ns}

Initialize an `MMatrix` of EPG states on CPU to be used throughout the simulation.
"""


@inline function initialize_states(::AbstractResource, sequence::EPGSimulator{T, Ns}) where {T,Ns}
    #Ω = @MMatrix zeros(Ω_eltype(sequence),3,Ns)
    #Ω = @SArray zeros(Ω_eltype(sequence),3,Ns,N)
    if nameof(typeof(sequence)) == ":FISP2DB"
        N = Int(ceil(sequence.H / (sequence.Vᵦ*sequence.TR)))
    else
        N = 1
    end 
    Ω = zeros(Ω_eltype(sequence),3,Ns,N)
end



"""
    initialize_states(::CUDALibs, sequence::EPGSimulator{T,Ns}) where {T,Ns}

Initialize an array of EPG states on a CUDA GPU to be used throughout the simulation.
"""
@inline function initialize_states(::CUDALibs, sequence::EPGSimulator{T,Ns}) where {T,Ns}
    # request shared memory in which configuration states are stored
    # (all threads request for the entire threadblock)
    Ω_shared = CUDA.CuStaticSharedArray(Ω_eltype(sequence), (3, Ns, THREADS_PER_BLOCK))
    # get view for configuration states of this thread's voxel
    # note that this function gets called inside a CUDA kernel
    # so it has has access to threadIdx
    Ω_view = view(Ω_shared,:,:,threadIdx().x)
    # wrap in a SizedMatrix
    Ω = SizedMatrix{3,Ns}(Ω_view)
    return Ω
end

# Initial conditions

"""
    initial_conditions!(Ω::EPGStates)

Set all components of all states to 0, except the Z-component of the 0th state which is set to 1.
"""
@inline function initial_conditions!(Ω::EPGStates)
    @. Ω = 0
    Z(Ω)[0,:] .= 1
    return nothing
end

"""
    full_blood_compensation!()

Set all F+ F- components to 0. This is used to simulate the effect of full blood compensation.
This function is used in the FISP2DB sequence. Only if the blood passes the slice completely. 
It is assumed that the signals from earlier blood are not measureable anymore and so the F+ and F- 
components can be set to 0.
"""
@inline function full_blood_compensation!(Ω::EPGStates)
    F₊(Ω) .= 0
    F̄₋(Ω) .= 0
    return nothing
end

# RF excitation

"""
    excite!(Ω::EPGStates, RF::Complex, p::AbstractTissueParameters)

Mixing of states due to RF pulse. Magnitude of RF is the flip angle in degrees.
Phase of RF is the phase of the pulse. If RF is real, the computations simplify a little bit.
"""
@inline function excite!(Ω::EPGStates, RF::T, p::AbstractTissueParameters) where T<:Union{Complex, Quantity{<:Complex}}

    # angle of RF pulse, convert from degrees to radians
    α = deg2rad(abs(RF))
    hasB₁(p) && (α *= p.B₁)
    # phase of RF pulse
    φ = angle(RF)

    x = α/2
    sinx, cosx = sincos(x)
    sin²x, cos²x = sinx^2, cosx^2
    # double angle formula
    sinα, cosα = 2*sinx*cosx, 2*cos²x - one(α)
    # phase stuff
    sinφ, cosφ = sincos(φ)
    # again double angle formula
    sin2φ, cos2φ = 2*sinφ*cosφ, 2*cosφ^2 - one(α)
    # complex exponentials
    ℯⁱᵠ  = complex(cosφ,  sinφ)
    ℯ²ⁱᵠ = complex(cos2φ, sin2φ)
    ℯ⁻ⁱᵠ  = conj(ℯⁱᵠ)
    ℯ⁻²ⁱᵠ = conj(ℯ²ⁱᵠ)
    # compute individual components of rotation matrix
    R₁₁, R₁₂, R₁₃ = cos²x, ℯ²ⁱᵠ * sin²x, -im * ℯⁱᵠ * sinα
    R₂₁, R₂₂, R₂₃ = ℯ⁻²ⁱᵠ * sin²x, cos²x, 1im * ℯ⁻ⁱᵠ * sinα #im gives issues with CUDA profiling, 1im works
    R₃₁, R₃₂, R₃₃ = -im * ℯ⁻ⁱᵠ * sinα / 2, 1im * ℯⁱᵠ * sinα / 2, cosα
    # assemble static matrix
    #R = SArray{Tuple{3,3}}(R₁₁,R₂₁,R₃₁,R₁₂,R₂₂,R₃₂,R₁₃,R₂₃,R₃₃)
    R = [R₁₁ R₁₂ R₁₃; R₂₁ R₂₂ R₂₃; R₃₁ R₃₂ R₃₃]
    # apply rotation matrix to each state
    for subvox = eachindex(Ω[1,1,:])
        Ωₛ = @view Ω[:,:,subvox]
        Ωₛ .= R * Ωₛ
    end
    return nothing
end
"""
    excite!(Ω::EPGStates, RF::T, p::AbstractTissueParameters) where T<:Union{Real, Quantity{<:Real}}

If RF is real, the calculations simplify (and probably Ω is real too, reducing memory (access) requirements).
"""
@inline function excite!(Ω::EPGStates, RF::T, p::AbstractTissueParameters) where T<:Union{Real, Quantity{<:Real}}

    # angle of RF pulse, convert from degrees to radians
    α = deg2rad(RF)
    hasB₁(p) && (α *= p.B₁)
    x = α/2
    sinx, cosx = sincos(x)
    sin²x, cos²x = sinx^2, cosx^2
    # double angle formula
    sinα, cosα = 2*sinx*cosx, 2*cos²x - one(α)
    # compute individual components of rotation matrix
    R₁₁, R₁₂, R₁₃ = cos²x, -sin²x, -sinα
    R₂₁, R₂₂, R₂₃ = -sin²x, cos²x, -sinα
    R₃₁, R₃₂, R₃₃ = sinα / 2, sinα / 2, cosα
    # assemble static matrix
    #R = SArray{Tuple{3,3}}(R₁₁,R₂₁,R₃₁,R₁₂,R₂₂,R₃₂,R₁₃,R₂₃,R₃₃)
    R = [R₁₁ R₁₂ R₁₃; R₂₁ R₂₂ R₂₃; R₃₁ R₃₂ R₃₃]
    # apply rotation matrix to each state
    for subvox = eachindex(Ω[1,1,:])
        Ωₛ = @view Ω[:,:,subvox]
        Ωₛ .= R * Ωₛ
    end
    return nothing
end

"""
    rotate!(Ω::EPGStates, eⁱᶿ::T) where T

Rotate `F₊` and `F̄₋` states under the influence of `eⁱᶿ = exp(i * ΔB₀ * Δt)`
"""
@inline function rotate!(Ω::EPGStates, eⁱᶿ::T) where T
    @. Ω[1:2,:,:] *= (eⁱᶿ,conj(eⁱᶿ))
end

# Decay

"""
    decay!(Ω::EPGStates, E₁, E₂)

T₂ decay for F-components, T₁ decay for `Z`-component of each state.
"""
@inline function decay!(Ω::EPGStates, E₁, E₂)
    @. Ω[:,:,:] *= (E₂, E₂, E₁)
end

"""
    rotate_decay!(Ω::EPGStates, E₁, E₂, eⁱᶿ)

Rotate and decay combined
"""
@inline function rotate_decay!(Ω::EPGStates, E₁, E₂, eⁱᶿ)
    @. Ω[:,:,:] *= (E₂*eⁱᶿ, E₂*conj(eⁱᶿ), E₁)
end

# Regrowth

"""
    regrowth!(Ω::EPGStates, E₁)

T₁ regrowth for Z-component of 0th order state.
"""
@inline function regrowth!(Ω::EPGStates, E₁)

    Z(Ω)[0,:] .+= (1 - E₁)
    #println(Z(Ω)[0])
end


# Dephasing

"""
    dephasing!(Ω::EPGStates)

Shift states around due to dephasing gradient:
The `F₊` go up one, the `F̄₋` go down one and `Z` do not change
"""
@inline function dephasing!(Ω::EPGStates)
    shift_down!(F̄₋(Ω))
    shift_up!(F₊(Ω), F̄₋(Ω))
end

# shift down the F- states, set highest state to 0
@inline function shift_down!(F̄₋)
    for j in 0:size(F̄₋,2)-1
        for i = 0:lastindex(F̄₋[:,j])-1
            @inbounds F̄₋[i,j] = F̄₋[i+1,j]
        end
    end
    for j in 0:size(F̄₋,2)-1
        @inbounds F̄₋[end,j] = 0
    end 
end

# shift up the F₊ states and let F₊[0] be conj(F₋[0])
@inline function shift_up!(F₊, F̄₋)
    for j in 0:size(F₊,2)-1
        for i = lastindex(F₊[:,0]):-1:1
            @inbounds F₊[i,j] = F₊[i-1,j]
        end
    end
    for j in 0:size(F₊,2)-1
        @inbounds F₊[0,j] = conj(F̄₋[0,j])
    end
end
# Invert

"""
    invert!(Ω::EPGStates, p::AbstractTissueParameters)

Invert `Z`-component of states of all orders. *Assumes fully spoiled transverse magnetization*.
"""
@inline function invert!(Ω::EPGStates, p::AbstractTissueParameters)
    # inversion angle
    θ = π
    hasB₁(p) && (θ *= p.B₁)
    Z(Ω) .*= cos(θ)
end

"""
    invert!(Ω::EPGStates)

Invert with B₁ insenstive (i.e. adiabatic) inversion pulse
"""
@inline function invert!(Ω::EPGStates)
    Z(Ω) .*= -1
end

# Spoil

"""
    spoil!(Ω::EPGStates)

Perfectly spoil the transverse components of all states.
"""
@inline function spoil!(Ω::EPGStates)
    F₊(Ω) .= 0
    F̄₋(Ω) .= 0
end

# Sample

"""
    sample_transverse!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)

Sample the measurable transverse magnetization, that is, the `F₊` component of the 0th state.
The `+=` is needed for 2D sequences where slice profile is taken into account.
"""
@inline function sample_transverse!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)
    @inbounds output[index] += F₊(Ω)[0]
end

@inline function sample_transverse_conj!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)
    @inbounds output[index] += F₋(Ω)[0]
end
"""
    sample_transverse"_V2!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)

Sample the measurable transverse magnetization, that is, the `F₊` component of the 0th state.
The `+=` is needed for 2D sequences where slice profile is taken into account.

For blood flow sequences all F₊ states from the 0th order are summed and scaled by the
partitioning amount (N)
"""
@inline function sample_transverse_V2!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)
    @inbounds output[index] += sum(F₊(Ω)[0,:])/size(Ω,3)
end

"""
    sample_Ω!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)

Sample the entire configuration state matrix `Ω`. The `+=` is needed
for 2D sequences where slice profile is taken into account.
"""
@inline function sample_Ω!(output, index::Union{Integer,CartesianIndex}, Ω::EPGStates)
    @inbounds output[index] .+= Ω
end

@inline function blood_shift!(Ω::EPGStates, z)
    z = min(z,1)
    for i = lastindex(Ω,3):-1:2
        @inbounds Ω[:,:,i] .= Ω[:,:,i-1] #Move dimensions forward
    end
    Ω[:,:,1] .= 0 #Initialize new dimension
    Ω[3,1,1] = z #Initialize Z component new dimension
end

