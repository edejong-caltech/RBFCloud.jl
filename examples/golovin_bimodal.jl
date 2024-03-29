""" collision-coalescence only with golovin kernel """

using Plots
using RBFCloud.BasisFunctions
using RBFCloud.MomentCollocation
using SpecialFunctions: gamma
using DifferentialEquations

function main()
    ############################ SETUP ###################################
    casename = "examples/golovin8_bimodal"

    # Numerical parameters
    FT = Float64
    tspan = (0.0, 4*3600.0)

    # basis setup 
    Nb = 8
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
    inject_rate = 0
    θ_r   = 3             # radius scale factor: µm
    k     = 3             # shape factor for particle size distribution 
  
    # initial/injection distribution in volume: gamma distribution in radius, number per cm^3
    #### INITIAL DISTRIBUTION: TWO MODES ####
    θ_v_1 = 1000.0
    N_1   = 10.0
    k_1   = 4
    θ_v_2 = 200.0
    N_2   = 100.0
    k_2   = 2
    r = v->(3/4/pi*v)^(1/3)
    mode1_v = v -> N_1*v^(k_1-1)/θ_v_1^k_1 * exp(-v/θ_v_1) / gamma(k_1)
    mode2_v = v -> N_2*v^(k_2-1)/θ_v_2^k_2 * exp(-v/θ_v_2) / gamma(k_2)
    n_v_init = v->(mode1_v(v) + mode2_v(v))
    
    n_v_inject = v -> (r(v))^(k-1)/θ_r^k * exp(-r(v)/θ_r) / gamma(k)
    
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
    println("means = ", rbf_loc)
    println("stddevs = ", rbf_shapes)
    rbf_loc = exp.(rbf_loc)

    # Injection rate
    function inject_rate_fn(v)
      f = inject_rate*n_v_inject(v)
      return f
    end
    ########################### PRECOMPUTATION ################################

    # Precomputation
    A = get_rbf_inner_products(basis, rbf_loc, tracked_moments)
    Source = get_kernel_rbf_source(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin)
    Sink = get_kernel_rbf_sink_precip(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin, xstop=vmax)
    (c_inject, Inject) = get_basis_projection(basis, rbf_loc, A, tracked_moments, inject_rate_fn, vmax)
    J = get_mass_cons_term(basis, xstart = vmin, xstop = vmax)

    # INITIAL CONDITION
    (c0, nj_init) = get_basis_projection(basis, rbf_loc, A, tracked_moments, n_v_init, vmax)
    println("precomputation complete")

    ########################### DYNAMICS ################################
    # Implicit Time stepping    
    function dndt(ni,t,p)
      return collision_coalescence(ni, A, Source, Sink, Inject)
    end

    prob = ODEProblem(dndt, nj_init, tspan)
    sol = solve(prob)
    t_coll = sol.t

    # track the moments
    basis_mom = vcat(get_moment(basis, 0.0, xstart=vmin, xstop=vmax)', get_moment(basis, 1.0, xstart=vmin, xstop=vmax)', get_moment(basis, 2.0, xstart=vmin, xstop=vmax)')
    c_coll = zeros(FT, length(t_coll), Nb)
    for (i,t) in enumerate(t_coll)
      nj_t = sol(t)
      c_coll[i,:] = get_constants_vec(nj_t, A)
    end
    mom_coll = c_coll*basis_mom'
    println("times = ", t_coll)
    println("c_init = ", c_coll[1,:])
    println("c_final = ", c_coll[end,:])

    # track the precipitable mass
    t_precip = collect(range(tspan[1], stop=tspan[2], length=100))
    m_precip = zeros(FT, length(t_precip))
    basis_mom_precip = get_moment(basis, 1.0, xstart=v_cutoff, xstop=vmax)
    for (i,t) in enumerate(t_precip)
      nj_t = sol(t)
      c_tmp = get_constants_vec(nj_t, A)
      m_precip[i] = c_tmp' * basis_mom_precip
    end

    #plot_nv_result(vmin*0.1, 1000.0, basis, c_coll[1,:], plot_exact=true, n_v_init=n_v_init, casename=casename)
    #plot_nv_result(vmin*0.1, 1000.0, basis, t_coll, c_coll, plot_exact=true, n_v_init=n_v_init, log_scale=true, casename = casename)
    #plot_nr_result(rmin*0.1, rmax, basis, t_coll, c_coll, plot_exact=true, n_v_init=n_v_init, log_scale=true, casename = casename)
    plot_moments(t_coll, mom_coll, casename = casename)
    plot_precip(t_precip, m_precip, v_cutoff, casename = casename)

    # output results to file
    open(string(casename,"results.txt"),"w") do file
      write(file, string("means = ", log.(rbf_loc),"\n"))
      write(file,string("stddevs = ", rbf_shapes,"\n"))
      write(file,"")
      write(file,string("times = ", t_coll,"\n"))
      write(file,string("M_0 = ", mom_coll[:,1],"\n"))
      write(file,string("M_1 = ", mom_coll[:,2],"\n"))
      write(file,string("M_2 = ", mom_coll[:,3],"\n"))
      write(file,"")
      write(file,string("c_init = ", c_coll[1,:],"\n"))
      write(file,string("c_final = ", c_coll[end,:],"\n"))
      write(file,"")
      write(file,string("t_precip = ", t_precip[:],"\n"))
      write(file,string("m_precip = ", m_precip[:],"\n"))
    end
end


""" Plot Initial distribution only """
function plot_init()
  r_plot = collect(range(0, stop=100.0, length=100))
  plot(r_plot, 
      n_v_init.(r_plot.^3*4*pi/3),
      linewidth=2,
      title="Initial distribution",
      ylabel="number /m^3 ",
      xlabel="r (µm)",
      xlim=[1, 100],
      ylim=[1e-2, 1e4],
      xaxis=:log
    )
  savefig("initial_dist.png")
end

""" Plot the n(v) result, with option to show exact I.C. and log or linear scale """
function plot_nv_result(vmin::FT, vmax::FT, basis::Array{CompactBasisFunc, 1}, t::Array{FT,1},
                        c::Array{FT, 2}; plot_exact::Bool=false, n_v_init::Function = x-> 0.0, 
                        log_scale::Bool=false, casename::String="") where {FT <: Real}
  v_plot = exp.(collect(range(log(vmin), stop=log(vmax), length=1000)))
  if plot_exact
    plot(v_plot,
        n_v_init.(v_plot),
        lw=2,
        label="Exact I.C.")
  else
    plot()
  end
  for (i,tsim) in enumerate(t)
    cvec = c[i,:]
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    if log_scale
      plot!(v_plot,
          n_plot,
          lw=2,
          ylim=[1e-2, 1e0],
          xlabel="volume, µm^3",
          ylabel="number",
          xaxis=:log,
          label=string("time ", tsim), legend=:topright)
    else
      plot!(v_plot,
          n_plot,
          lw=2,
          ylim=[1e-2, 1e0],
          xlabel="volume, µm^3",
          ylabel="number",
          label=string("time ", tsim), legend=:topright)
    end
  end

  savefig(string(casename,"nv.png"))
end


""" Plot the n(r) result, with option to show exact I.C. and log or linear scale """
function plot_nr_result(rmin::FT, rmax::FT, basis::Array{CompactBasisFunc, 1}, t::Array{FT,1}, c::Array{FT, 2};
                        plot_exact::Bool=false, n_v_init::Function = x-> 0.0, 
                        log_scale::Bool=false, casename::String="") where {FT <: Real}
  r_plot = exp.(collect(range(log(rmin), stop=log(rmax), length=1000)))
  v_plot = 4/3*pi*r_plot.^3
  if plot_exact
    plot(r_plot,
          n_v_init.(v_plot),
          lw=2,
          label="Exact")
  end
  for (i, tsim) in enumerate(t)
    cvec = c[i,:]
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    if log_scale
      plot!(r_plot,
            n_plot,
            lw=2,
            xlabel="radius, µm",
            ylabel="number",
            xaxis=:log,
            ylim=[1e-2, 1e0], legend=:topright)
    else
      plot!(r_plot,
            n_plot,
            lw=2,
            xlabel="radius, µm",
            ylabel="number",
            ylim=[1e-2, 1e0], legend=:topright)
    end
  end
  savefig(string(casename,"nr.png"))
end

""" Plot the moments supplied over time """
function plot_moments(tsteps::Array{FT}, moments::Array{FT, 2}; casename::String="") where {FT <: Real}
  plot(tsteps,
        moments[:,1],
        lw=2,
        xlabel="time, sec",
        ylabel="number / cm^3",
        label="M_0")
  for i=1:length(moments[1,:])
    plot(tsteps, 
          moments[:,i],
          lw=2,
          xlabel="time, sec",
          ylabel=string("M_",i-1),
          label=string("M_",i-1))
    savefig(string(casename,"M_",i-1,".png"))
  end
end

""" Plot the precipitable portion supplied over time """
function plot_precip(t_precip::Array{FT}, m_precip::Array{FT}, v_cutoff::FT; casename::String="") where {FT <: Real}
  plot(t_precip,
        m_precip,
        lw=2,
        xlabel="time, sec",
        ylabel="μm^3 / cm^3",
        label="precip mass")
  savefig(string(casename,"precip_",v_cutoff,".png"))
end

@time main()
