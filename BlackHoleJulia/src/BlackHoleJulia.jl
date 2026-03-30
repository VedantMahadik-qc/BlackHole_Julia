module BlackHoleJulia

include("metric.jl")
include("geodesic.jl")
include("raytracer.jl")
include("render.jl")
include("gpu_raytracer.jl")    # ← ADD THIS

using .Metric, .Geodesic, .RayTracer, .Render, .GPURayTracer

export render_image, render_gpu, save_render

end