include("../src/BlackHoleJulia.jl")
using .BlackHoleJulia
using CUDA
using Plots

println("GPU: ", CUDA.name(CUDA.device()))
println("Rendering wallpaper...")

# 4K resolution — your RTX 4070 can handle it!
# Render square first, then embed in 16:9 canvas
@time img = render_gpu(3840, 2160, 1.0f0; cam_dist=20.0f0)   # square render

# Pad with black on sides to make 16:9
using Images

heatmap(Float64.(img),
    color=:inferno,
    axis=false,
    border=:none,
    colorbar=false,
    background_color=:black,
    foreground_color=:black,
    size=(3840, 2160),
    left_margin=(-8, :mm),
    right_margin=(-8, :mm),
    top_margin=(-8, :mm),
    bottom_margin=(-8, :mm),
    dpi=200
)
mkpath(joinpath(@__DIR__, "..", "output"))
savefig(joinpath(@__DIR__, "..", "output", "blackhole_wallpaper_4K.png"))
println("Done! Saved to output/blackhole_wallpaper_4K.png")