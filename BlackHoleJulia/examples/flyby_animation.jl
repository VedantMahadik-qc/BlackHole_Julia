include("../src/BlackHoleJulia.jl")
using .BlackHoleJulia
using Plots

function main()
    M   = 1.0
    r_s = 2.0 * M

    function step_particle(pos, vel, M, dt=0.05)
        r     = sqrt(pos[1]^2 + pos[2]^2)
        accel = -M .* pos ./ r^3
        vel   = vel .+ accel .* dt
        pos   = pos .+ vel   .* dt
        return pos, vel
    end

    # Scenarios
    r_orb   = 6.0
    v_orb   = sqrt(M / r_orb)
    pos1 = [r_orb, 0.0];  vel1 = [0.0, v_orb]
    pos2 = [10.0, -6.0];  vel2 = [-0.25, 0.18]
    pos3 = [8.0, 0.0];    vel3 = [0.0, sqrt(M / 8.0) * 0.65]

    n_steps   = 800
    orb_trail = Vector{Vector{Float64}}()
    fly_trail = Vector{Vector{Float64}}()
    inf_trail = Vector{Vector{Float64}}()

    for _ in 1:n_steps
        push!(orb_trail, copy(pos1))
        push!(fly_trail, copy(pos2))
        push!(inf_trail, copy(pos3))

        r1 = sqrt(pos1[1]^2 + pos1[2]^2)
        r2 = sqrt(pos2[1]^2 + pos2[2]^2)
        r3 = sqrt(pos3[1]^2 + pos3[2]^2)

        if r1 > r_s; pos1, vel1 = step_particle(pos1, vel1, M) end
        if r2 > r_s; pos2, vel2 = step_particle(pos2, vel2, M) end
        if r3 > r_s; pos3, vel3 = step_particle(pos3, vel3, M) end
    end

    θ     = range(0, 2π, length=200)
    trail = 80

    println("Rendering animation frames...")

    anim = @animate for k in 1:4:n_steps
        plot(bg=:black, fg=:white,
             xlims=(-12, 12), ylims=(-12, 12),
             aspect_ratio=:equal,
             size=(800, 800),
             legend=:topright,
             title="Black Hole Encounters",
             titlefontcolor=:white,
             axis=false, border=:none)

        # Add this just before the event horizon plot:
        for glow_r in [3.5, 3.0, 2.5]
        plot!(glow_r .* cos.(θ), glow_r .* sin.(θ),
          color=:orange, alpha=0.08, lw=8, label=false)
        end
        plot!(r_s .* cos.(θ), r_s .* sin.(θ),
              color=:white, lw=2, label="Event Horizon", fill=true, fillcolor=:black)

        plot!(range(-8, 8, length=100), zeros(100),
              color=:orange, alpha=0.3, lw=1, label=false)

        scatter!([0.0], [0.0], color=:white, ms=8, label=false)

        t_start = max(1, k - trail)

        xs = [orb_trail[i][1] for i in t_start:k]
        ys = [orb_trail[i][2] for i in t_start:k]
        plot!(xs, ys, color=:cyan, lw=1.5, alpha=0.6, label=false)
        scatter!([orb_trail[k][1]], [orb_trail[k][2]],
                 color=:cyan, ms=6, label="Stable Orbit")

        xs = [fly_trail[i][1] for i in t_start:k]
        ys = [fly_trail[i][2] for i in t_start:k]
        plot!(xs, ys, color=:lime, lw=1.5, alpha=0.6, label=false)
        scatter!([fly_trail[k][1]], [fly_trail[k][2]],
                 color=:lime, ms=6, label="Flyby")

        xs = [inf_trail[i][1] for i in t_start:k]
        ys = [inf_trail[i][2] for i in t_start:k]
        plot!(xs, ys, color=:orange, lw=1.5, alpha=0.6, label=false)
        scatter!([inf_trail[k][1]], [inf_trail[k][2]],
                 color=:orange, ms=6, label="Spiral Infall")
    end

    println("Saving GIF...")
    mkpath(joinpath(@__DIR__, "..", "output"))
    gif(anim, joinpath(@__DIR__, "..", "output", "flyby_animation.gif", fps=30))
    println("Done! Saved to output/flyby_animation.gif")
end

main()