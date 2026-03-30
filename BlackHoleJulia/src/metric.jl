module Metric

using LinearAlgebra
export schwarzschild_force

function schwarzschild_force(pos::Vector{Float64}, vel::Vector{Float64}, M::Float64)
    r = norm(pos)
    # GR photon deflection = 2x Newtonian (first-order GR correction)
    accel = -2.0 * M .* pos ./ r^3
    return accel
end

end