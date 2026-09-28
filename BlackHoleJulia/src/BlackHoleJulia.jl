module BlackHoleJulia

include("metric.jl")
include("geodesic.jl")
include("raytracer.jl")
include("render.jl")
include("gpu_raytracer.jl")
include("kerr_gpu_raytracer.jl")    # ← spin / frame dragging, GPU-only for now
include("kerr_orbit.jl")            # ← orbiting camera, for flybys

using .Metric, .Geodesic, .RayTracer, .Render, .GPURayTracer, .KerrGPURayTracer, .KerrOrbit

export render_image, render_gpu, save_render, render_kerr_gpu, render_kerr_orbit_frame

end