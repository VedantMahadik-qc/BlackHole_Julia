module KerrGPURayTracer

using KernelAbstractions
using CUDA

export render_kerr_gpu

@inline function star_brightness(dx::Float32, dy::Float32, dz::Float32)
    ix = Int32(floor(dx * 800.0f0)) * Int32(73856093)
    iy = Int32(floor(dy * 800.0f0)) * Int32(19349663)
    iz = Int32(floor(dz * 800.0f0)) * Int32(83492791)
    h  = xor(ix, iy, iz)
    return (mod(abs(h), Int32(250)) < Int32(1)) ? 1.0f0 : 0.0f0
end

@kernel function kerr_kernel!(img, M, a, width, height, cam_dist)
    i, j = @index(Global, NTuple)

    r_s        = 2.0f0 * M
    disk_inner = r_s * 1.5f0
    disk_outer = 8.0f0
    fov        = Float32(π / 3.0)
    aspect     = width / height

    px = (2.0f0 * (i - 0.5f0) / width  - 1.0f0) * tan(fov/2.0f0) * aspect
    py = (2.0f0 * (j - 0.5f0) / height - 1.0f0) * tan(fov/2.0f0)

    dx, dy, dz = -1.0f0, px, py
    len = sqrt(dx^2 + dy^2 + dz^2)
    dx /= len; dy /= len; dz /= len

    x, y, z    = cam_dist, 0.0f0, 0.5f0
    vx, vy, vz = dx, dy, dz

    pixel_val = 0.0f0
    dt        = 0.05f0

    for _ in 1:2000
        r   = sqrt(x^2 + y^2 + z^2)
        rho² = r^2 + a^2 * z^2 / (r^2 + 1f-6)

        # Kerr event horizon radius
        r_kerr = M + sqrt(M^2 - a^2)

        if r <= r_kerr * 1.05f0
            pixel_val = 0.0f0
            break
        end

        new_z = z + vz * dt
        if z * new_z < 0.0f0
            r_cross = sqrt(x^2 + y^2)
            if disk_inner < r_cross < disk_outer
                brightness = 1.0f0 - (r_cross - disk_inner)/(disk_outer - disk_inner)
                # Kerr Doppler — stronger asymmetry than Schwarzschild
                doppler    = 1.0f0 + 0.8f0 * a * (y / (r_cross + 1f-6))
                pixel_val  = clamp(0.5f0 + 0.6f0 * brightness * doppler, 0.0f0, 1.0f0)
                break
            end
        end

        # Kerr geodesic force
        ax = -2.0f0 * M * x / rho²^1.5f0
        ay = -2.0f0 * M * y / rho²^1.5f0
        az = -2.0f0 * M * z / rho²^1.5f0

        # Frame dragging
        ω   = 2.0f0 * M * a * r / (rho²^2 + a^2 * r^2 + 1f-6)
        ax += ω * vy
        ay -= ω * vx

        vx += ax * dt; vy += ay * dt; vz += az * dt
        x  += vx * dt; y  += vy * dt; z   = new_z
    end

    # Stars
    if pixel_val == 0.0f0
        len2 = sqrt(vx^2 + vy^2 + vz^2) + 1f-6
        pixel_val = star_brightness(vx/len2, vy/len2, vz/len2) * 0.9f0
    end

    img[j, i] = pixel_val
end

function render_kerr_gpu(width::Int, height::Int,
                          M::Float32=1.0f0, a::Float32=0.9f0;
                          cam_dist=20.0f0)
    backend = CUDABackend()
    img     = CUDA.zeros(Float32, height, width)

    kernel! = kerr_kernel!(backend, (16, 16))
    kernel!(img, M, a, Float32(width), Float32(height), Float32(cam_dist),
            ndrange=(width, height))

    CUDA.synchronize()
    return Array(img)
end

end