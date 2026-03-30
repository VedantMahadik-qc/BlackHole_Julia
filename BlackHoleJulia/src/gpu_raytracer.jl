module GPURayTracer

using KernelAbstractions
using CUDA
using LinearAlgebra

export render_gpu

@kernel function raytrace_kernel!(img, M, width, height, cam_dist)
    i, j = @index(Global, NTuple)

    r_s        = 2.0f0 * M
    disk_inner = r_s * 1.5f0
    disk_outer = 8.0f0
    fov        = Float32(π / 3.0)

    px = (2.0f0 * (i - 0.5f0) / width  - 1.0f0) * tan(fov / 2.0f0)
    py = (2.0f0 * (j - 0.5f0) / height - 1.0f0) * tan(fov / 2.0f0)

    dx, dy, dz = -1.0f0, px, py
    len = sqrt(dx^2 + dy^2 + dz^2)
    dx /= len; dy /= len; dz /= len

    x, y, z    = cam_dist, 0.0f0, 0.5f0
    vx, vy, vz = dx, dy, dz

    pixel_val = 0.0f0
    dt        = 0.05f0

    for _ in 1:2000
        r = sqrt(x^2 + y^2 + z^2)

        if r <= r_s * 1.1f0
            pixel_val = 0.0f0
            break
        end

        new_z = z + vz * dt
        if z * new_z < 0.0f0
            r_cross = sqrt(x^2 + y^2)
            if disk_inner < r_cross < disk_outer
                brightness = 1.0f0 - (r_cross - disk_inner) / (disk_outer - disk_inner)
                doppler    = 1.0f0 + 0.5f0 * (y / r_cross)
                pixel_val  = clamp(0.5f0 + 0.5f0 * brightness * doppler, 0.0f0, 1.0f0)
                break
            end
        end

        ax = -2.0f0 * M * x / r^3
        ay = -2.0f0 * M * y / r^3
        az = -2.0f0 * M * z / r^3

        vx += ax * dt; vy += ay * dt; vz += az * dt
        x  += vx * dt; y  += vy * dt; z   = new_z
    end

    img[j, i] = pixel_val
end

function render_gpu(width::Int, height::Int, M::Float32=1.0f0; cam_dist=15.0f0)
    backend = CUDABackend()
    img     = CUDA.zeros(Float32, height, width)

    kernel! = raytrace_kernel!(backend, (16, 16))
    kernel!(img, M, Float32(width), Float32(height), Float32(cam_dist),
            ndrange=(width, height))

    CUDA.synchronize()
    return Array(img)
end

end