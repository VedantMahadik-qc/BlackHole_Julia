module KerrMetric

export kerr_force

"""
Geodesic acceleration for a photon in Kerr spacetime.
Uses Boyer-Lindquist coordinates simplified to Cartesian.
a = spin parameter (0 to 1, where 1 = maximally spinning)
M = black hole mass
"""
function kerr_force(pos::Vector{Float32}, vel::Vector{Float32},
                    M::Float32, a::Float32)
    x, y, z = pos[1], pos[2], pos[3]
    r²  = x^2 + y^2 + z^2
    r   = sqrt(r²)

    # Kerr parameter ρ²
    rho² = r² + a^2 * z^2 / r²

    # Kerr effective potentials — frame dragging terms
    ax = -M * x / rho²^1.5f0 - a^2 * M * x * z^2 / (r² * rho²^1.5f0)
    ay = -M * y / rho²^1.5f0 - a^2 * M * y * z^2 / (r² * rho²^1.5f0)
    az = -M * z / rho²^1.5f0 + a^2 * M * z   / (r  * rho²^1.5f0)

    # Frame dragging — rotational coupling (key Kerr effect!)
    ω  = 2.0f0 * M * a * r / (rho²^2 + a^2 * r²)
    ax += ω * vel[2]    # couples y-velocity to x-force
    ay -= ω * vel[1]    # couples x-velocity to y-force

    return [ax, ay, az]
end

end