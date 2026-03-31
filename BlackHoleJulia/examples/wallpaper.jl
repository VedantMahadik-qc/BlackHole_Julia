using CUDA
using KernelAbstractions
using Images: colorview, RGB
using FileIO

@kernel function wallpaper_kernel!(img_r, img_g, img_b, M, a, width, height, cam_z)
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

    pr = 0.0f0; pg = 0.0f0; pb = 0.0f0
    hit = false

    for _ in 1:2000
        r    = sqrt(x^2 + y^2 + z^2)
        rho² = r^2 + a^2 * z^2 / (r^2 + 1f-6)

        if r <= r_kerr * 1.05f0
            hit = true; break
        end

        new_z = z + vz * dt
        if z * new_z < 0.0f0
            r_cross = sqrt(x^2 + y^2)
            if disk_inner < r_cross < disk_outer
                t       = 1.0f0 - (r_cross - disk_inner) / (disk_outer - disk_inner)
                doppler = 1.0f0 + 0.8f0 * a * (y / (r_cross + 1f-6))
                bright  = clamp(sqrt(t) * doppler * 1.2f0, 0.0f0, 1.0f0)

                pr = clamp(bright * 1.0f0, 0.0f0, 1.0f0)
                pg = clamp(bright * (0.15f0 + 0.85f0 * t^1.2f0), 0.0f0, 1.0f0)
                pb = clamp(bright * t^1.8f0, 0.0f0, 1.0f0)
                hit = true; break
            end
        end

        ax_  = -2.0f0 * M * x / rho²^1.5f0
        ay_  = -2.0f0 * M * y / rho²^1.5f0
        az_  = -2.0f0 * M * z / rho²^1.5f0
        ω    = 2.0f0 * M * a * r / (rho²^2 + a^2 * r^2 + 1f-6)
        ax_ += ω * vy; ay_ -= ω * vx

        vx += ax_ * dt; vy += ay_ * dt; vz += az_ * dt
        x  += vx  * dt; y  += vy  * dt; z   = new_z
    end

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
            b  = 0.6f0 + 0.4f0 * Float32(mod(abs(h), Int32(100))) / 100.0f0
            st = mod(abs(h), Int32(10))
            if st < Int32(3)
                pr = b*0.8f0; pg = b*0.9f0; pb = b
            elseif st < Int32(7)
                pr = b; pg = b; pb = b*0.95f0
            else
                pr = b; pg = b*0.5f0; pb = b*0.2f0
            end
        end
    end

    img_r[j, i] = pr
    img_g[j, i] = pg
    img_b[j, i] = pb
end

function render_wallpaper(W, H, a, cam_z)
    backend = CUDABackend()
    r_gpu   = CUDA.zeros(Float32, H, W)
    g_gpu   = CUDA.zeros(Float32, H, W)
    b_gpu   = CUDA.zeros(Float32, H, W)
    kernel! = wallpaper_kernel!(backend, (16, 16))
    kernel!(r_gpu, g_gpu, b_gpu, 1.0f0, a,
            Float32(W), Float32(H), cam_z,
            ndrange=(W, H))
    CUDA.synchronize()
    return colorview(RGB, permutedims(
        cat(Array(r_gpu), Array(g_gpu), Array(b_gpu), dims=3), (3,1,2)))
end

# ── SETTINGS — tweak these ───────────────────────────────────────
W      = 3840          # width  (3840 = 4K, 2560 = 1440p, 3440 = ultrawide)
H      = 2160          # height (2160 = 4K, 1440 = 1440p)
a      = 0.9f0        # spin — 0.99 = near-max, most dramatic
cam_z  = 0.5f0         # elevation — 0.0 = edge-on ring, 1.5 = top-down

println("GPU: ", CUDA.name(CUDA.device()))
println("Rendering $(W)×$(H) wallpaper  (a=$(a), cam_z=$(cam_z))...")
@time img = render_wallpaper(W, H, a, cam_z)

mkpath("output")
out = "output/wallpaper_$(W)x$(H)_a$(a).png"
save(out, img)
println("✅ Done → $out")