module RayTracer

using ..Geodesic: trace_ray
using LinearAlgebra

export render_image

function render_image(width::Int, height::Int, M::Float64=1.0; cam_dist=15.0)
    img   = zeros(Float64, height, width)
    r_s   = 2.0 * M
    disk_inner = r_s * 1.5
    disk_outer = 8.0

    cam_pos = [cam_dist, 0.0, 0.5]   # slight vertical offset for disk visibility
    fov     = π / 4.0                 # 45 degree FOV

    for j in 1:height
        for i in 1:width
            px  = (2.0 * (i - 0.5) / width  - 1.0) * tan(fov / 2)
            py  = (2.0 * (j - 0.5) / height - 1.0) * tan(fov / 2)
            dir = [-1.0, px, py]

            sol = trace_ray(cam_pos, dir, M)

            # Check each step of the ray path for crossings
            hit_hole = false
            hit_disk = false

            for k in 2:length(sol.u)
                prev = sol.u[k-1][1:3]
                curr = sol.u[k][1:3]

                # Hit event horizon?
                if norm(curr) <= r_s * 1.1
                    hit_hole = true
                    break
                end

                # Crossed accretion disk plane (z sign change)?
                if prev[3] * curr[3] < 0.0
                    r_cross = norm(curr[1:2])
                    if disk_inner < r_cross < disk_outer
                        hit_disk = true
                        doppler = 1.0 + 0.5 * (curr[2] / r_cross)
                        brightness = (1.0 - (r_cross - disk_inner) / (disk_outer - disk_inner)) * doppler
                        img[j, i] = clamp(0.6 + 0.4 * brightness, 0.0, 1.0)
                        break
                    end
                end
            end

            if hit_hole
                img[j, i] = 0.0        # Black hole → pure black
            elseif !hit_disk
                img[j, i] = 0.15       # Background → dark
            end
        end
    end

    return img
end

end