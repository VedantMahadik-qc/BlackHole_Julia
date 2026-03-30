module GPURayTracer

using KernelAbstractions
using LinearAlgebra

export render_gpu

# ── GPU KERNEL ─────────────────────────────────────────────────────────────────
# Each GPU thread handles ONE pixel — this is the key speedup
@kernel function raytrace_kernel!(img, M, cam_pos, fov, width, height)
    i, j = @index(Global, NTuple)   # each thread = one pixel

    r_s        = 2.0f0 * M
    disk_inner = r_s * 1.5f0
    disk_outer = 8.0f0

    px  = (2.0f0 * (i - 0.5f0) / width  - 1.0f0) * tan(fov / 2.0f0)
    py  = (2.0f0 * (j - 0.5f0) / height - 1.0f0) * tan(fov / 2.0f0)

    # Ray direction
    dx, dy, dz = -1.0f0, px, py
    len = sqrt(dx^2 + dy^2 + dz^2)
    dx /= len; dy /= len; dz /= len

    # Ray position
    x, y, z   = cam_pos[1], cam_pos[2], cam_pos[3]
    vx, vy, vz = dx, dy, dz

    pixel_val = 0.15f0   # default: dark background

    # Integrate geodesic (100 steps inline — no ODE solver on GPU)
    dt = 0.05f0
    for _ in 1:2000
        r = sqrt(x^2 + y^2 + z^2)

        if r <= r_s * 1.1f0
            pixel_val = 0.0f0   # swallowed by black hole
            break
        end

        # Check disk crossing (z sign change)
        new_z = z + vz * dt
        if z * new_z < 0.0f0
            r_cross = sqrt(x^2 + y^2)
            if disk_inner < r_cross < disk_outer
                brightness = 1.0f0 - (r_cross - disk_inner) / (disk_outer - disk_inner)
                doppler    = 1.0f0 + 0.5f0 * (y / r_cross)
                pixel_val  = clamp(0.6f0 + 0.4f0 * brightness * doppler, 0.0f0, 1.0f0)
                break
            end
        end

        # GR force
        ax = -2.0f0 * M * x / r^3
        ay = -2.0f0 * M * y / r^3
        az = -2.0f0 * M * z / r^3

        vx += ax * dt; vy += ay * dt; vz += az * dt
        x  += vx * dt; y  += vy * dt; z  += new_z - z
        z   = new_z
    end

    img[j, i] = pixel_val
end

# ── LAUNCH FUNCTION ────────────────────────────────────────────────────────────
function render_gpu(width::Int, height::Int, M::Float32=1.0f0; cam_dist=15.0f0)
    backend = get_backend()   # auto-detects CUDA / AMD / CPU

    img     = KernelAbstractions.zeros(backend, Float32, height, width)
    cam_pos = KernelAbstractions.ones(backend, Float32, 3)
    cam_pos[1] = cam_dist; cam_pos[2] = 0.5f0; cam_pos[3] = 0.5f0

    kernel! = raytrace_kernel!(backend, (16, 16))   # 16x16 thread blocks
    kernel!(img, M, cam_pos, Float32(π/4), Float32(width), Float32(height),
            ndrange=(width, height))

    KernelAbstractions.synchronize(backend)
    return Array(img)   # copy back from GPU to CPU for saving
end

end