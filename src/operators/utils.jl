# Rotate the x,y-components of a 3D vector with angle θ.
@inline function rotxy(sinθ, cosθ, (x,y,z)::Isochromat{T}) where T
    u = cosθ * x - sinθ * y
    v = cosθ * y + sinθ * x
    w = z
    return Isochromat{T}(u,v,w)
end

@inline function off_resonance_rotation(state::EPGStates, Δt::T, p::AbstractTissueParameters) where {S<:Union{Isochromat,EPGStates},T}
    if hasB₀(p)
        θ = π*Δt*p.B₀*2
        return exp(im*θ)
    else
        return one(T)
    end
end

# state not used within this function but it's needed for manual ad to dispatch on the state
@inline E₁(state::EPGStates, Δt, T₁) where S<:Union{Isochromat,EPGStates} = exp(-Δt * inv(T₁))
@inline E₂(state::EPGStates, Δt, T₂) where S<:Union{Isochromat,EPGStates} = exp(-Δt * inv(T₂))

