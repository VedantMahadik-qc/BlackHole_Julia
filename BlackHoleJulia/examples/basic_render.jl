
include("../src/BlackHoleJulia.jl")
using .BlackHoleJulia

println("Tracing rays...")
img = render_image(400, 400, 1.0)

println("Saving image...")
save_render(img, "output/blackhole.png")