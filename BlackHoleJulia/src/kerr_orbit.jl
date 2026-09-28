module KerrOrbit

using KernelAbstractions
using CUDA

export render_kerr_orbit_frame

# Same Kerr physics as kerr_gpu_raytracer.jl's kernel (event horizon,
# frame dragging, accretion disk with Doppler asymmetry) -- the only
# difference is the camera orbits at `cam_angle` around the z-axis and
# always re-aims at the origin, instead of sitting at a fixed position
# looking down a fixed axis. kerr_gpu_raytracer.jl is untouched.

@inline function star_brightness(dx::Float32, dy::Float32, dz::Float32)
    ix = Int32(floor(dx * 800.0f0)) * Int32(73856093)
    iy = Int32(floor(dy * 800.0f0)) * Int32(19349663)
    iz = Int32(floor(dz * 800.0f0)) * Int32(83492791)
    h  = xor(ix, iy, iz)
    return (mod(abs(h), Int32(250)) < Int32(1)) ? 1.0f0 : 0.0f0
end

@kernel function orbit_kernel!(img, M, a, width, height, cam_dist, cam_angle)
    i, j = @index(Global, NTuple)

    r_s        = 2.0f0 * M
    disk_inner = r_s * 1.5f0
    disk_outer = 8.0f0
    fov        = Float32(π / 3.0)
    aspect     = width / height

    # Camera orbits in the xy-plane at fixed radius and height, angle
    # sweeps a full circle as cam_angle goes 0 -> 2π across frames.
    cx = cam_dist * cos(cam_angle)
    cy = cam_dist * sin(cam_angle)
    cz = 0.5f0

    # Forward = normalized vector from camera toward the origin
    fx, fy, fz = -cx, -cy, -cz
    flen = sqrt(fx^2 + fy^2 + fz^2) + 1f-6
    fx /= flen; fy /= flen; fz /= flen

    # Proper look-at camera basis: right = forward × world-up,
    # then true-up = right × forward (Gram-Schmidt against world-up).
    upx, upy, upz = 0.0f0, 0.0f0, 1.0f0
    rx = fy*upz - fz*upy
    ry = fz*upx - fx*upz
    rz = fx*upy - fy*upx
    rlen = sqrt(rx^2 + ry^2 + rz^2) + 1f-6
    rx /= rlen; ry /= rlen; rz /= rlen

    uxr = ry*fz - rz*fy
    uyr = rz*fx - rx*fz
    uzr = rx*fy - ry*fx

    px = (2.0f0 * (i - 0.5f0) / width  - 1.0f0) * tan(fov / 2.0f0) * aspect
    py = (2.0f0 * (j - 0.5f0) / height - 1.0f0) * tan(fov / 2.0f0)

    dx = fx + px*rx + py*uxr
    dy = fy + px*ry + py*uyr
    dz = fz + px*rz + py*uzr
    dlen = sqrt(dx^2 + dy^2 + dz^2) + 1f-6
    dx /= dlen; dy /= dlen; dz /= dlen

    x, y, z    = cx, cy, cz
    vx, vy, vz = dx, dy, dz

    pixel_val = 0.0f0
    dt        = 0.05f0

    for _ in 1:2000
        r    = sqrt(x^2 + y^2 + z^2)
        rho² = r^2 + a^2 * z^2 / (r^2 + 1f-6)
        r_kerr = M + sqrt(M^2 - a^2)

        if r <= r_kerr * 1.05f0
            pixel_val = 0.0f0
            break
        end

        new_z = z + vz * dt
        if z * new_z < 0.0f0
            r_cross = sqrt(x^2 + y^2)
            if disk_inner < r_cross < disk_outer
                brightness = 1.0f0 - (r_cross - disk_inner) / (disk_outer - disk_inner)
                doppler    = 1.0f0 + 0.8f0 * a * (y / (r_cross + 1f-6))
                pixel_val  = clamp(0.5f0 + 0.6f0 * brightness * doppler, 0.0f0, 1.0f0)
                break
            end
        end

        ax = -2.0f0 * M * x / rho²^1.5f0
        ay = -2.0f0 * M * y / rho²^1.5f0
        az = -2.0f0 * M * z / rho²^1.5f0

        ω   = 2.0f0 * M * a * r / (rho²^2 + a^2 * r^2 + 1f-6)
        ax += ω * vy
        ay -= ω * vx

        vx += ax * dt; vy += ay * dt; vz += az * dt
        x  += vx * dt; y  += vy * dt; z   = new_z
    end

    if pixel_val == 0.0f0
        len2 = sqrt(vx^2 + vy^2 + vz^2) + 1f-6
        pixel_val = star_brightness(vx/len2, vy/len2, vz/len2) * 0.9f0
    end

    img[j, i] = pixel_val
end

function render_kerr_orbit_frame(width::Int, height::Int, M::Float32, a::Float32,
                                  cam_dist::Float32, cam_angle::Float32)
    backend = CUDABackend()
    img     = CUDA.zeros(Float32, height, width)

    kernel! = orbit_kernel!(backend, (16, 16))
    kernel!(img, M, a, Float32(width), Float32(height), cam_dist, cam_angle,
            ndrange=(width, height))

    CUDA.synchronize()
    return Array(img)
end

end
