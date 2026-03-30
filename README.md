# BlackHole_Julia 🌌

A physically-based **black hole ray tracer** written in Julia, 
GPU-accelerated via CUDA.jl on an NVIDIA RTX 4070 Laptop GPU.

Built from scratch in one evening — no external physics libraries.

---

## Features
- ⚡ **GPU-accelerated** ray tracing via `KernelAbstractions.jl` + `CUDA.jl`
- 🔭 **Schwarzschild metric** — geodesic light ray integration
- 💡 **Gravitational lensing** — accretion disk light bends over event horizon
- 🌟 **Lensed star field** — background stars warp around the black hole
- 🟠 **Doppler brightening** — accretion disk asymmetry simulation
- 🎬 **Particle animation** — orbit, flyby, and spiral infall scenarios
- 🖼️ **4K wallpaper output** in ~12 seconds on RTX 4070

---

## Output

### 4K Wallpaper Render
*(Upload your best wallpaper PNG here via GitHub UI)*

### Particle Flyby Animation
*(Upload flyby_animation.gif here via GitHub UI)*

---

## How to Run

### Requirements
- Julia 1.10+
- NVIDIA GPU with CUDA (for GPU render)

### Install Dependencies
```julia
julia --project=.
] add OrdinaryDiffEq Plots KernelAbstractions CUDA
```

### CPU Render (any machine)
```bash
julia --project=. --threads=auto BlackHoleJulia/examples/basic_render.jl
```

### GPU Render (NVIDIA only)
```bash
julia --project=. BlackHoleJulia/examples/gpu_render.jl
```

### Particle Animation
```bash
julia --project=. --threads=auto BlackHoleJulia/examples/flyby_animation.jl
```

---

## Physics

Light rays follow **geodesics** in Schwarzschild spacetime. 
The geodesic equation in geometric units (G=c=1):

d²xᵢ/dλ² = -2M/r³ · xᵢ

Rays that cross the **Schwarzschild radius** `r_s = 2M` are absorbed. 
Rays crossing the accretion disk plane within `[1.5·r_s, 8M]` are lit 
with Doppler-shifted brightness.

---

## Inspired By
- Kavan's black hole simulation in C++
- Kip Thorne's scientific work on Interstellar's Gargantua
- Event Horizon Telescope imaging of M87*

---

*Built with Julia 1.12 · Pop!_OS · RTX 4070 Laptop GPU*