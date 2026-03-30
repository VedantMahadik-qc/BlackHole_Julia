using CUDA
using KernelAbstractions
using Images
using FileIO

@kernel function kerr_rgb_kernel!(img_r, img_g, img_b, M, a, width, height, cam_z)
    i, j = @index(Global, NTuple)

    fov    = Float32(π / 3.0)
    aspect = width / height

    px = (2.0f0 * (i - 0.5f0) / width  - 1.0f0) * tan(fov/2.0f0) * aspect
    py = (2.0f0 * (j - 0.5f0) / height - 1.0f0) * tan(fov/2.0f0)

    dx, dy, dz = -1.0f0, px, py
    len = sqrt(dx^2 + dy^2 + dz^2)
    dx /= len; dy /= len; dz /= len

    x, y, z    = 20.0f0, 0.0f0, cam_z
    vx, vy, vz = dx, dy, dz

    r_kerr     = M + sqrt(M^2 - a^2)
    disk_inner = r_kerr * 1.5f0
    disk_outer = 12.0f0
    dt         = 0.05f0

    hit_r = 0.0f0
    hit_g = 0.0f0
    hit_b = 0.0f0
    hit   = false

    for _ in 1:2000
        r    = sqrt(x^2 + y^2 + z^2)
        rho² = r^2 + a^2 * z^2 / (r^2 + 1f-6)

        if r <= r_kerr * 1.05f0
            hit = true
            hit_r = 0.0f0; hit_g = 0.0f0; hit_b = 0.0f0
            break
        end

        new_z = z + vz * dt
        if z * new_z < 0.0f0
            r_cross = sqrt(x^2 + y^2)
            if disk_inner < r_cross < disk_outer
                t       = 1.0f0 - (r_cross - disk_inner) / (disk_outer - disk_inner)
                doppler = 1.0f0 + 0.8f0 * a * (y / (r_cross + 1f-6))
                bright  = clamp(sqrt(t) * doppler * 1.2f0, 0.0f0, 1.0f0)

# Colour temperature uses t directly (NOT multiplied by bright again)
                cr = 1.0f0
                cg = 0.15f0 + 0.85f0 * t^1.2f0          # white at t=1, orange at t=0.5, red at t=0
                cb = t^1.8f0    # only appears near inner hot edge

                hit_r = clamp(bright * cr, 0.0f0, 1.0f0)
                hit_g = clamp(bright * cg, 0.0f0, 1.0f0)
                hit_b = clamp(bright * cb, 0.0f0, 1.0f0)

                hit = true
                break
            end
        end

        ax  = -2.0f0 * M * x / rho²^1.5f0
        ay  = -2.0f0 * M * y / rho²^1.5f0
        az  = -2.0f0 * M * z / rho²^1.5f0
        ω   = 2.0f0 * M * a * r / (rho²^2 + a^2 * r^2 + 1f-6)
        ax += ω * vy
        ay -= ω * vx

        vx += ax * dt; vy += ay * dt; vz += az * dt
        x  += vx * dt; y  += vy * dt; z   = new_z
    end

    # Stars with colour temperature
    if !hit
        len2 = sqrt(vx^2 + vy^2 + vz^2) + 1f-6
        nx = vx/len2; ny = vy/len2; nz = vz/len2

        ix = Int32(floor(nx * 2000.0f0)) * Int32(1664525)
        iy = Int32(floor(ny * 2000.0f0)) * Int32(1013904223)
        iz = Int32(floor(nz * 2000.0f0)) * Int32(22695477)
        h  = xor(ix + iy, iz)
        h  = h * Int32(1664525) + Int32(1013904223)
        h  = xor(h, h >> Int32(16))

        if mod(abs(h), Int32(300)) < Int32(2)
            brightness = 0.6f0 + 0.4f0 * Float32(mod(abs(h), Int32(100))) / 100.0f0
            # Star colour: use h to pick blue/white/orange/red
            star_type = mod(abs(h), Int32(10))
            if star_type < Int32(3)        # blue-white (hot)
                hit_r = brightness * 0.8f0
                hit_g = brightness * 0.9f0
                hit_b = brightness * 1.0f0
            elseif star_type < Int32(7)    # white (sun-like)
                hit_r = brightness
                hit_g = brightness
                hit_b = brightness * 0.95f0
            else                           # orange-red (cool)
                hit_r = brightness * 1.0f0
                hit_g = brightness * 0.6f0
                hit_b = brightness * 0.3f0
            end
        end
    end

    img_r[j, i] = hit_r
    img_g[j, i] = hit_g
    img_b[j, i] = hit_b
end

function render_rgb(width, height, M=1.0f0, a=0.9f0, cam_z=0.5f0)
    backend = CUDABackend()
    r_gpu   = CUDA.zeros(Float32, height, width)
    g_gpu   = CUDA.zeros(Float32, height, width)
    b_gpu   = CUDA.zeros(Float32, height, width)

    kernel! = kerr_rgb_kernel!(backend, (16, 16))
    kernel!(r_gpu, g_gpu, b_gpu, M, a, Float32(width), Float32(height), Float32(cam_z),
            ndrange=(width, height))
    CUDA.synchronize()

    R = Array(r_gpu)
    G = Array(g_gpu)
    B = Array(b_gpu)

    # Stack into RGB image for Images.jl
    img = colorview(RGB, R, G, B)
    return img
end

println("GPU: ", CUDA.name(CUDA.device()))
println("Rendering RGB Kerr black hole...")

@time img = render_rgb(3840, 2160, 1.0f0, 0.9f0, 0.5f0)

mkpath("output")
save("output/kerr_rgb_4K.png", img)
println("Saved → output/kerr_rgb_4K.png")