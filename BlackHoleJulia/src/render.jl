module Render

using Plots

export save_render

function save_render(img::Matrix{<:Real}, filename="output/blackhole.png")
    mkpath(dirname(filename))   # ← ADD THIS LINE — creates folder automatically
    heatmap(img,
        color=:inferno,
        axis=false,
        border=:none,
        size=(800, 600),
        title="Black Hole Simulation — Julia"
    )
    savefig(filename)
    println("Saved to $filename")
end
end