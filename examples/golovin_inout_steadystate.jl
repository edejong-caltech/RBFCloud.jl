"""Solve for steady-state collision-coalescence only with golovin kernel """

using Plots
using RBFCloud.BasisFunctions
using RBFCloud.MomentCollocation
using SpecialFunctions: gamma
using DifferentialEquations
using NonNegLeastSquares
using LinearAlgebra

function main()
    ############################ SETUP ###################################
    casename = "examples/steadystate_golovin16_inject_remove_"

    # Numerical parameters
    FT = Float64
    #tspan = (0.0, 2*3600.0)
    c_final = [64.77288882940162, 20.267835856622163, 256.1327187494738, 286.615562971068, 77.12373065102274, 24.775584462821683, 6.935543884762892, 10.81230051432805, 3.9963657263359105, 0.0, 0.0, 1.4464180466510988e-5, 0.0, 0.0, 0.0, 0.0]

    # basis setup 
    Nb = 64
    rmax  = 200.0
    rmin  = 2.0
    vmin = 4/3*pi*rmin^3
    vmax = 4/3*pi*rmax^3

    r_cutoff = 25
    v_cutoff = 4/3*pi*r_cutoff^3

    # Physical parameters: Kernel
    a = 0.0
    b = 1500 * 1e-12
    c = 0.0    
    r = v->(3/4/pi*v)^(1/3)
    area = v->4*pi*r(v)^2
    kernel_func = x -> a + b*(x[1]+x[2]) + c*(r(x[1])+r(x[2]))^2*abs(area(x[1])-area(x[2]))
    tracked_moments = [1.0]
    inject_rate = 1.0
    N     = 0             # initial droplet density: number per cm^3
    θ_v   = 1000           # volume scale factor: µm^3
    θ_v_in= 200           # volume scale factor: μm
    k     = 2             # shape factor for particle size distribution 

    # initial/injection distribution in volume: gamma distribution in radius, number per cm^3
    r = v->(3/4/pi*v)^(1/3)
    n_v_init = v -> N*v^(k-1)/θ_v^k * exp(-v/θ_v) / gamma(k)
    n_v_inject = v -> v^(k-1)/θ_v_in^k * exp(-v/θ_v_in) / gamma(k)
    
    # lin-spaced log compact rbf
    basis = Array{CompactBasisFunc}(undef, Nb)
    rbf_loc = collect(range(log(vmin), stop=log(vmax), length=Nb))
    rbf_shapes = zeros(Nb)
    rbf_shapes[3:end] = (rbf_loc[3:end] - rbf_loc[1:end-2])
    rbf_shapes[2] = rbf_loc[2]
    rbf_shapes[1] = rbf_loc[2]
    for i=1:Nb
      basis[i] = CompactBasisFunctionLog(rbf_loc[i], rbf_shapes[i])
    end
    # println("means = ", rbf_loc)
    # println("stddevs = ", rbf_shapes)
    rbf_loc = exp.(rbf_loc)

    # basis = Array{CompactBasisFunc}(undef, Nb)
    # rbf_loc = collect(range(log(vmin), stop=log(vmax), length=Nb))
    # rbf_loc = exp.(rbf_loc)
    # rbf_shapesL = zeros(Nb)
    # rbf_shapesR = zeros(Nb)
    # rbf_shapesL[2:end] = ((rbf_loc[2:end] - rbf_loc[1:end-1]))*0.5
    # rbf_shapesR[1:end-1] = rbf_shapesL[2:end]
    # rbf_shapesL[1] = rbf_loc[1]
    # rbf_shapesR[end] = rbf_shapesR[end-1]
    # # rbf_shapes[2] = rbf_loc[2]
    # # rbf_shapes[1] = rbf_loc[2]*0.5
    # # println("means = ", rbf_loc)
    # # println("sL = ", rbf_shapesL)
    # # println("sR = ", rbf_shapesR)
    # for i=1:Nb
    #   basis[i] = PiecewiseConstant(rbf_loc[i], rbf_shapesL[i], rbf_shapesR[i])
    # end

    # Injection rate
    function inject_rate_fn(v)
      f = inject_rate*n_v_inject(v)
      return f
    end
    ########################### PRECOMPUTATION ################################

    # Precomputation
    A = get_rbf_inner_products(basis, rbf_loc, tracked_moments, x_stop=v_cutoff)
    Source = get_kernel_rbf_source(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin, xstop=v_cutoff)
    Sink = get_kernel_rbf_sink_precip(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin, xstop=v_cutoff)
    (c_inject, Inject) = get_basis_projection(basis, rbf_loc, A, tracked_moments, inject_rate_fn, v_cutoff)
    #J = get_mass_cons_term(basis, xstart = vmin, xstop = vmax)

    # INITIAL CONDITION
    # (c0, nj_init) = get_basis_projection(basis, rbf_loc, A, tracked_moments, n_v_init, vmax)
    println("precomputation complete")

    Q = 1/2*Source - Sink 
    #r_in = nonneg_lsq(Matrix(I, Nb, Nb),c_inject)[:,1]
    #r_in = c_inject
    r_in = Inject

    ########################### NEWTONS METHOD ################################
    # Implicit Time stepping    
    function Jacobian(c)
        Jc = zeros(FT, Nb, Nb)
        for l=1:Nb
            for j=1:Nb
                Jc[l,j] = c'*Q[l,j,:] + c'*Q[l,:,j]
            end
        end
        return Jc
    end

    function zero_this_fn(c)
        fc = zeros(FT, Nb)
        for j=1:Nb
            fc[j] = c'*Q[j,:,:]*c + r_in[j]
        end
        return fc
    end

    function newton_step(c, fc)
        Jc = Jacobian(c)
        #c_new = c - nonneg_lsq(Jc, fc)[:,1]
        c_new = c - Jc\fc
        c_new = nonneg_lsq(Matrix(I,Nb,Nb), c_new)[:,1]
        fc_new = zero_this_fn(c_new)
        return (c_new, fc_new)
    end

    #c_guess = c_final
    c_guess = c_inject*10

    basis_mom_withsink = vcat(get_moment(basis, 0.0, xstart=vmin, xstop=v_cutoff)', get_moment(basis, 1.0, xstart=vmin, xstop=v_cutoff)', get_moment(basis, 2.0, xstart=vmin, xstop=v_cutoff)')

    fc_guess = zero_this_fn(c_guess)
    Jc_guess = Jacobian(c_guess)
    println(c_guess)
    println(sum(fc_guess))
    #println(Jc_guess)

    c_prev = c_guess
    fc_prev = fc_guess
    for step=1:20
        (c_next, fc_next) = newton_step(c_prev, fc_prev)
        c_prev = c_next
        fc_prev = fc_next
        println(sum(fc_prev), c_prev'*basis_mom_withsink')
    end

    println()
    println(c_guess)
    println(c_prev)

    open(string(casename,"results.txt"),"w") do file
        write(file, string("means = ", rbf_loc,"\n"))
        write(file,string("stddevs = ", rbf_shapes,"\n"))
        write(file,string("c_final = ", c_prev,"\n"))
    end
end

@time main()