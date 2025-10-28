using ArgParse
using GLMakie

struct SimParams{T<:AbstractFloat}
    Du::T
    Dv::T
    F::T
    k::T
end

function default_sim_params()
    SimParams(0.16, 0.08, 0.0367, 0.0649)
end

function grey_scott_step!(U::AbstractMatrix{T}, V::AbstractMatrix{T}, params::SimParams{T}, dt::T) where T<:AbstractFloat
    Du, Dv, F, k = params.Du, params.Dv, params.F, params.k
    U_new = copy(U)
    V_new = copy(V)
    Nx, Ny = size(U)
    for i in 1:Nx
        for j in 1:Ny
            # Compute Laplacians with periodic boundary conditions
            ileft, iright = mod1(i-1,Nx), mod1(i+1,Nx)
            jtop, jbottom = mod1(j-1,Ny), mod1(j+1,Ny)
            lap_U = U[iright,j] + U[ileft,j] + U[i,jbottom] + U[i,jtop] - 4*U[i,j]
            lap_V = V[iright,j] + V[ileft,j] + V[i,jbottom] + V[i,jtop] - 4*V[i,j]
            # Reaction terms
            uvv = U[i,j]*V[i,j]^2
            U_new[i,j] = U[i,j] + (Du * lap_U - uvv + F * (1 - U[i,j])) * dt
            V_new[i,j] = V[i,j] + (Dv * lap_V + uvv - (F + k) * V[i,j]) * dt
        end
    end
    U .= U_new
    V .= V_new
end

struct IdxHelper{T<:Integer}
    left::Vector{T}
    right::Vector{T}
    top::Vector{T}
    bottom::Vector{T}
end

function IdxHelper(Nx::Integer, Ny::Integer)
    left  = [mod1(i-1,Nx) for i in 1:Nx]
    right = [mod1(i+1,Nx) for i in 1:Nx]
    top   = [mod1(j-1,Ny) for j in 1:Ny]
    bottom= [mod1(j+1,Ny) for j in 1:Ny]
    IdxHelper(left, right, top, bottom)
end

function laplace!(L::AbstractMatrix{T}, M::AbstractMatrix{T}, idxh::IdxHelper{Int}) where T<:AbstractFloat
    Nx, Ny = size(M)
    @inbounds for i in 1:Nx
        il = idxh.left[i]
        ir = idxh.right[i]
        for j in 1:Ny    
            jt = idxh.top[j]
            jb = idxh.bottom[j]
            L[i,j] = M[ir,j] + M[il,j] + M[i,jb] + M[i,jt] - 4*M[i,j]
        end
    end
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nx"
            help = "Number of grid points in x direction"
            arg_type = Int
            default = 100
        "--ny"
            help = "Number of grid points in y direction"
            arg_type = Int
            default = 100
        "--ns"
            help = "Seed square size"
            arg_type = Int
            default = 10
        "--rand"
            help = "Randomize initial condition"
            arg_type = Float32
            default = 0.02
        "--du"
            help = "Diffusion coefficient for U"
            arg_type = Float32
            default = 0.16
        "--dv"
            help = "Diffusion coefficient for V"
            arg_type = Float32
            default = 0.08
        "--f"
            help = "Feed rate"
            arg_type = Float32
            default = 0.0367
        "--k"
            help = "Kill rate"
            arg_type = Float32
            default = 0.0649
        "--dt"
            help = "Time step"
            arg_type = Float32
            default = 1.0
        "--steps"
            help = "Number of simulation steps"
            arg_type = Union{Int,Nothing}
            default = nothing
        "--fps"
            help = "Frames per second for visualization"
            arg_type = Int
            default = 1
    end
    parse_args(s)
end

function initialize_UV(args)
    Nx, Ny, Ns = args["nx"], args["ny"], args["ns"]
    rnd = args["rand"]

    U =  ones(Float32, Nx, Ny)
    V = zeros(Float32, Nx, Ny)
    
    # Initial condition: small square in the center
    cx, cy = div(Nx,2), div(Ny,2)
    U[cx-Ns:cx+Ns, cy-Ns:cy+Ns] .= 0.50f0
    V[cx-Ns:cx+Ns, cy-Ns:cy+Ns] .= 0.25f0
    
    # Add some random noise
    if rnd > 0.0
        U .+= rnd * (rand(Float32, Nx, Ny) .- 0.5f0)
        V .+= rnd * (rand(Float32, Nx, Ny) .- 0.5f0)
    end
    U, V
end

function show_until_closed(fig)
    display(fig)
    while isopen(fig.scene)
        yield()     # give time to the render/event loop
        sleep(0.05) # small nap to avoid busy-wait
    end
end

function simulation(args)
    dt = args["dt"]
    nsteps = args["steps"]
    frame_ns = 1e9 / args["fps"]
    params = SimParams(args["du"], args["dv"], args["f"], args["k"])

    U, V = initialize_UV(args)
    
    set_theme!(theme_black())
    fig = Figure(size = (900, 900))
    ax  = Axis(fig[1, 1], title = "Gray-Scott (F=$(params.F), k=$(params.k))", xlabel="x", ylabel="y")
    Vshow = copy(V)
    data  = Observable(Vshow)
    hm = heatmap!(ax, data; interpolate=false, colorrange=(0.0, 1.0))
    Colorbar(fig[1, 2], hm, label = "V")
    screen = display(fig)

    frame_no = 1
    while isopen(screen)    
        t = time_ns()
        grey_scott_step!(U, V, params, dt)
        st = (time_ns() - t) / 1e3
        ax.title[] = "Frame $frame_no step time: $(round(st, digits=2)) us"
        @inbounds Vshow .= V   # zaktualizuj bufor wizualizacji bez alokacji
        notify(data)           # poinformuj Makie, że dane się zmieniły
        yield()                # oddaj sterowanie, by Makie mógł odrysować
        tsleep = (frame_ns - (time_ns() - t)) / 1e9
        tsleep < 0 || sleep(tsleep)
        frame_no += 1
        if nsteps !== nothing && frame_no > nsteps
            yield()
            break
        end
    end
end


GLMakie.activate!()
println("running grey-scott reaction-diffusion simulation...")
args = parse_commandline()

simulation(args)
