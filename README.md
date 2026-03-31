# BlackHole.jl 

> GPU-accelerated Kerr black hole raytracer written in Julia.  
> Renders a physically accurate spinning black hole with relativistic 
> accretion disk in seconds on consumer hardware.

![Kerr Black Hole](output/wallpaper_3840x2160_a0.9.png)

---

## ✨ Features

- **Kerr metric geodesic raytracing** — full spinning black hole physics
- **Frame dragging** — spacetime rotation via spin parameter `a` (0→1)
- **Relativistic Doppler boosting** — asymmetric disk brightness from spin
- **RGB blackbody accretion disk** — white-hot inner edge → deep crimson outer
- **4K GPU rendering** — ~10 seconds on RTX 4070 via CUDA.jl
- **Live orbital viewer** — drag to orbit the black hole in real time (GLMakie)
- **Coloured starfield** — blue/white/red stars by temperature class

---

## 🚀 Quick Start

### Requirements
- Julia 1.10+
- NVIDIA GPU with CUDA support
- ~2GB disk space for Julia packages

### Install

```bash
git clone https://github.com/VedantMahadik-qc/BlackHole_Julia
cd BlackHole_Julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Static 4K Render

```bash
julia --project=. BlackHoleJulia/examples/kerr_render.jl
# Output → output/wallpaper_3840x2160.png
```

### Interactive Orbital Viewer

```bash
julia --project=. BlackHoleJulia/examples/kerr_interactive.jl
# Drag mouse to orbit the black hole in real time
```

---

## 🎛️ Parameters

| Parameter | Description | Default |
|---|---|---|
| `a` | Spin (0 = Schwarzschild, 0.99 = near-max Kerr) | `0.99` |
| `M` | Black hole mass (geometric units) | `1.0` |
| `cam_z` | Camera elevation above disk plane | `0.8` |
| `disk_outer` | Accretion disk outer radius | `14.0` |
| `W, H` | Output resolution | `3840×2160` |

---

## 🔭 Physics

Geodesics are integrated using a simplified Boyer-Lindquist form of the 
Kerr metric. The frame dragging term couples photon velocity components 
via the angular momentum parameter `ω = 2Mar / (ρ⁴ + a²r²)`.  
Accretion disk colour follows a blackbody temperature gradient — inner 
orbits (high `t`) render white-hot, outer orbits cool to deep crimson.

---

## 📸 Gallery

| Gargantua style (z=0.5) | M87* style (z=0.0) | Ultimate (a=0.99) |
|---|---|---|
| *(add your renders)* | *(add your renders)* | *(add your renders)* |

---

## 🛠️ Built With

- [Julia](https://julialang.org/) 
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
- [GLMakie.jl](https://github.com/MakieOrg/Makie.jl)
- [Images.jl](https://github.com/JuliaImages/Images.jl)