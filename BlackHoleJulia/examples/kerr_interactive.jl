using GLMakie
using CUDA
using KernelAbstractions
using Images: colorview, RGB

@kernel function kerr_orbit_kernel!(img, M, a, cam_x, cam_y, cam_z, width, height)
    i, j = @index(Global, NTuple)

    fov    = Float32(π / 3.0)
    aspect = width / height

    px = (2.0f0 * (i - 0.5f0) / width  - 1.0f0) * tan(fov/2.0f0) * aspect
    py = (2.0f0 * (j - 0.5f0) / height - 1.0f0) * tan(fov/2.0f0)

    # Camera looks toward origin (black hole)
    forward_x = -cam_x; forward_y = -cam_y; forward_z = -cam_z
    flen = sqrt(forward_x^2 + forward_y^2 + forward_z^2)
    forward_x /= flen; forward_y /= flen; forward_z /= flen

    # Right vector = forward × up
    up_x = 0.0f0; up_y = 0.0f0; up_z = 1.0f0
    right_x = forward_y*up_z - forward_z*up_y
    right_y = forward_z*up_x - forward_x*up_z
    right_z = forward_x*up_y - forward_y*up_x
    rlen = sqrt(right_x^2 + right_y^2 + right_z^2) + 1f-6
    right_x /= rlen; right_y /= rlen; right_z /= rlen

    # True up = right × forward
    tup_x = right_y*(-forward_z) - right_z*(-forward_y)
    tup_y = right_z*(-forward_x) - right_x*(-forward_z)
    tup_z = right_x*(-forward_y) - right_y*(-forward_x)

    # Ray direction
    dx = forward_x + px*right_x + py*tup_x
    dy = forward_y + px*right_y + py*tup_y
    dz = forward_z + px*right_z + py*tup_z
    dlen = sqrt(dx^2 + dy^2 + dz^2)
    dx /= dlen; dy /= dlen; dz /= dlen

    x, y, z    = cam_x, cam_y, cam_z
    vx, vy, vz = dx, dy, dz

    r_kerr     = M + sqrt(M^2 - a^2)
    disk_inner = r_kerr * 1.5f0
    disk_outer = 12.0f0
    dt         = 0.05f0

    pr = 0.0f0; pg = 0.0f0; pb = 0.0f0
    hit = false

    for _ in 1:1500
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

    img[j, i, 1] = pr
    img[j, i, 2] = pg
    img[j, i, 3] = pb
end

function launch_orbit(width=800, height=500, a=0.9f0, M=1.0f0)
    backend  = CUDABackend()
    gpu_img  = CUDA.zeros(Float32, height, width, 3)

    # Spherical camera — azimuth and elevation
    cam_dist = Ref(20.0f0)
    azimuth  = Ref(0.0f0)      # left/right orbit
    elevation= Ref(0.3f0)      # up/down tilt (radians)

    function cam_pos()
        az = azimuth[]; el = elevation[]
        cx = cam_dist[] * cos(el) * cos(az)
        cy = cam_dist[] * cos(el) * sin(az)
        cz = cam_dist[] * sin(el)
        return Float32(cx), Float32(cy), Float32(cz)
    end

    function rerender()
        cx, cy, cz = cam_pos()
        kernel! = kerr_orbit_kernel!(backend, (16, 16))
        kernel!(gpu_img, Float32(M), Float32(a), cx, cy, cz,
                Float32(width), Float32(height),
                ndrange=(width, height))
        CUDA.synchronize()
        raw = Array(gpu_img)
        return colorview(RGB, permutedims(raw, (3,2,1)))
    end

    cpu_img = Observable(rerender())

    fig = Figure(size=(width, height), backgroundcolor=:black)
    ax  = Axis(fig[1,1], aspect=DataAspect(), backgroundcolor=:black)
    deregister_interaction!(ax, :rectanglezoom)
    deregister_interaction!(ax, :scrollzoom)
    deregister_interaction!(ax, :limitreset)
    hidedecorations!(ax)
    hidespines!(ax)
    ax.title = "Kerr BH  |  Drag to orbit  |  a=$(a)"
    ax.titlecolor = :white
    image!(ax, cpu_img)

    dragging   = Ref(false)
    drag_start = Ref(Point2f(0.0f0, 0.0f0))
    az_start   = Ref(0.0f0)
    el_start   = Ref(0.3f0)

    on(events(fig).mousebutton) do event
        if event.button == Mouse.left
            if event.action == Mouse.press
                dragging[]   = true
                drag_start[] = Point2f(events(fig).mouseposition[])
                az_start[]   = azimuth[]
                el_start[]   = elevation[]
            elseif event.action == Mouse.release
                dragging[] = false
            end
        end
    end

    on(events(fig).mouseposition) do pos
        if dragging[]
            delta      = Point2f(pos) - drag_start[]
            azimuth[]  = az_start[]  - delta[1] * 0.01f0   # left/right
            elevation[]= clamp(el_start[] + delta[2] * 0.01f0,
                               -1.4f0, 1.4f0)              # up/down clamped
            cpu_img[]  = rerender()
        end
    end

    display(fig)
    println("🕳️  Drag to orbit the black hole!")
    println("    Left/right = azimuth, Up/down = elevation")
    wait(fig.scene)
end

launch_orbit(800, 500, 0.9f0)