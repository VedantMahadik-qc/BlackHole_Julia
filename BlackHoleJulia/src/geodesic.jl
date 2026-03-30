module Geodesic

using OrdinaryDiffEq
using LinearAlgebra
using ..Metric: schwarzschild_force

export trace_ray

function trace_ray(pos0::Vector{Float64}, dir0::Vector{Float64}, M::Float64;
                   tspan=(0.0, 100.0), dt=0.02)
    r_s = 2.0 * M

    function geodesic_ode!(du, u, p, t)
        pos = u[1:3]
        vel = u[4:6]
        r   = norm(pos)

        if r <= r_s
            du .= 0.0
            return
        end

        accel = schwarzschild_force(pos, vel, M)
        du[1:3] = vel
        du[4:6] = accel
    end

    u0   = vcat(pos0, normalize(dir0))
    prob = ODEProblem(geodesic_ode!, u0, tspan)
    sol  = solve(prob, RK4(), dt=dt, adaptive=false)
    return sol
end

end