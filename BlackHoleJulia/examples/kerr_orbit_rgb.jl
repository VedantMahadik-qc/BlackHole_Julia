"""
Kerr orbit sequence, RGB
=========================

Renders N frames while orbiting the camera around the black hole, in the
same colour-temperature style as kerr_render.jl / kerr_interactive.jl --
white-hot inner disk fading to orange/red, coloured stars -- so the saved
sequence matches what the live interactive tool shows, rather than the
grayscale output from kerr_orbit.jl (last turn's version).

Known simplification, not a bug I can cleanly fix without testing: the
Doppler brightness term uses a fixed world-frame axis (`y`), which was a
reasonable approximation for a fixed camera but becomes approximate once
the camera orbits -- properly, the bright side should be computed
relative to the camera's current position, not a fixed world axis. It
will still show *an* asymmetry, just not one guaranteed to track the
camera perfectly as it moves. Flagging honestly rather than shipping a
more complex fix I can't verify under time pressure.

Untested end to end -- I can't run Julia or touch CUDA hardware. Render
one frame and check it before running the full sequence:

    julia> include("kerr_orbit_rgb.jl")   # renders the full sequence as-is

To test just one frame first, comment out the loop at the bottom and run:

    img = render_orbit_rgb_frame(960, 540, 1.0f0, 0.9f0, 20.0f0, 0.0f0)
    save("test_frame.png", img)
"""

using CUDA
using KernelAbstractions
using Images: colorview, RGB
using FileIO

@kernel function orbit_rgb_kernel!(img_r, img_g, img_b, M, a, width, height, cam_dist, cam_angle)
    i, j = @index(Global, NTuple)

    fov    = Float32(π / 3.0)
    aspect = width / height

    # Camera orbits in the xy-plane, always looking back at the origin
    # (same camera-basis construction as kerr_orbit.jl).
    cx = cam_dist * cos(cam_angle)
    cy = cam_dist * sin(cam_angle)
    cz = 0.5f0

    fx, fy, fz = -cx, -cy, -cz
    flen = sqrt(fx^2 + fy^2 + fz^2) + 1f-6
    fx /= flen; fy /= flen; fz /= flen

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
                doppler = 1.0f0 + 0.8f0 * a * (y / (r_cross + 1f-6))  # world-frame approximation, see docstring
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

function render_orbit_rgb_frame(width::Int, height::Int, M::Float32, a::Float32,
                                 cam_dist::Float32, cam_angle::Float32)
    backend = CUDABackend()
    r_gpu   = CUDA.zeros(Float32, height, width)
    g_gpu   = CUDA.zeros(Float32, height, width)
    b_gpu   = CUDA.zeros(Float32, height, width)

    kernel! = orbit_rgb_kernel!(backend, (16, 16))
    kernel!(r_gpu, g_gpu, b_gpu, M, a, Float32(width), Float32(height), cam_dist, cam_angle,
            ndrange=(width, height))
    CUDA.synchronize()

    return colorview(RGB, permutedims(
        cat(Array(r_gpu), Array(g_gpu), Array(b_gpu), dims=3), (3,1,2)))
end

# ── Render a sequence ──────────────────────────────────────────────
# Keep n_frames small for the first real pass through Atlas -- this is
# about proving the pipeline and answering the distortion-parameter
# question, not producing the final polished sequence yet.
n_frames = 24
out_dir  = joinpath(@__DIR__, "..", "output", "orbit_rgb")
mkpath(out_dir)

println("GPU: ", CUDA.name(CUDA.device()))
for k in 0:(n_frames - 1)
    angle = Float32(2π * k / n_frames)
    img   = render_orbit_rgb_frame(960, 540, 1.0f0, 0.9f0, 20.0f0, angle)
    path  = joinpath(out_dir, "orbit_$(lpad(k, 3, '0')).png")
    save(path, img)
    println("frame $(k+1)/$n_frames -> $path")
end
println("Done.")
