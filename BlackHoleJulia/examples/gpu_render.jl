include("../src/BlackHoleJulia.jl")
using .BlackHoleJulia
using CUDA
using Plots

println("GPU: ", CUDA.name(CUDA.device()))
println("Rendering wallpaper...")

# 4K resolution — your RTX 4070 can handle it!
# Render square first, then embed in 16:9 canvas
@time img = render_gpu(2160, 2160, 1.0f0; cam_dist=20.0f0)   # square render

# Pad with black on sides to make 16:9
using Images
black_pad = zeros(Float32, 2160, 840)   # (3840-2160)/2 = 840 each side
img_wide  = hcat(black_pad, img, black_pad)   # 3840 × 2160

heatmap(Float64.(img_wide),
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
mkpath("output")
savefig("output/blackhole_wallpaper_4K.png")
println("Done! Saved to output/blackhole_wallpaper_4K.png")