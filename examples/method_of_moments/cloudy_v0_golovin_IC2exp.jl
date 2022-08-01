"Linear coalescence kernel example"

using DifferentialEquations
using Plots
using QuadGK

using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources


function main()
  ############################ SETUP ###################################
  casename = "examples/method_of_moments/golovin_IC2exp_"
  r_cutoff = 25
  v_cutoff = 4/3*pi*r_cutoff^3

  # Numerical parameters
  FT = Float64
  tspan = (0.0, 2*3600.0)
  tol = 1e-8

  rmax  = 200.0
  rmin  = 2.0
  vmin = 4/3*pi*rmin^3
  vmax = 4/3*pi*rmax^3

  # Physical parameters: Kernel
  a = 0.0
  b = 1500 * 1e-12
  c = 0.0
  kernel_func = x -> a + b*(x[1]+x[2]) + c*abs(x[1]^(2/3)-x[2]^(2/3))/vmax^(2/3)*(x[1]^(1/3)+x[2]^(1/3))^2
  kernel = CoalescenceTensor(kernel_func, 1, 100.0)

  # Initial condition
  N     = 10.0            # initial droplet density: number per cm^3
  θ_r   = 15.0            # radius scale factor: μm
  θ_v   = 4/3*pi*θ_r^3    # volume scale factor: µm
  k     = 1.0             # shape factor for particle size distribution 
  moments_init = [N, N*θ_v]
  dist_init = ExponentialPrimitiveParticleDistribution(N, θ_v)

  open(string(casename,"results.txt"),"w") do file
    write(file,string("initial N, k, theta = ", N,", ", k,", ", θ_v,", ","\n"))
  end
  # Set up the ODE problem
  # Step 1) Define termination criterion: stop integration when one of the 
  #         distribution parameters leaves its allowed domain (which can 
  #         happen before the end of the time period defined below by tspan)
  """
  function out_of_bounds(m, t, integrator, lbound, ubound)

  - `m` - unknown function of the ODE; moments as a function of t
  - `t` - ODE time
  - `integrator` - ODE integrator (see DifferentialEquations.jl docs)
                   Its user-provided data (integrator.p) has to be a
                   Dict with a key ":dist", whose value is the 
                   ParticleDistribution at the previous time step.
  - `lbound` - the lower bounds of the parameters. Default to -Inf if left
               unspecified.
  - `ubound` - the upper bounds of the parameters. Default to Inf if left
               unspecified.

  Returns true if one or more of the distribution parameters are outside their
  bounds. Returns false otherwise.
  """
  function out_of_bounds(m, t, integrator; 
                         lbound::Union{AbstractVector, Nothing}=nothing, 
                         ubound::Union{AbstractVector, Nothing}=nothing)

    dist = integrator.p[:dist]
    dist_params = reduce(vcat, ParticleDistributions.get_params(dist)[2])
    n_params = length(dist_params)

    lbound == nothing && (lbound = -Inf * ones(n_params))
    ubound == nothing && (ubound = Inf * ones(n_params))

    if sum(lbound .<= dist_params .<= ubound) < n_params
      println("Exiting integration at time t=$(integrator.t)")
      println("At least one of the distribution parameters has left the ",
              "allowed domain")
      return true
    else
      return false
    end
  end

  # Define bounds: n ≧ 1.0, θ > 0.0, k > 0.0)
  lbound = [1.0, 1e-5]
  #lbound = [1.0, 1e-5, 1e-5]
  condition(m, t, integrator) = out_of_bounds(m, t, integrator; lbound=lbound)
  affect!(integrator) = terminate!(integrator)
  cb = DiscreteCallback(condition, affect!)

  # Step 2) Set up the right hand side of ODE
  rhs(m, par, t) = get_int_coalescence(m, par, kernel)

  # Make the initial distribution a parameter of the ODE, so that it can get 
  # updated and carried along over the entire integration time.
  ODE_parameters = Dict(:dist => dist_init)
  prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
  sol = solve(prob, Tsit5(), callback=cb, reltol=tol, abstol=tol)

#   Plot the solution for the 0th, 1st and 2nd moment
#   pyplot()
  time = vcat(sol.t[:]'...)[:]
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  moment_2 = 2 * moment_1.^2 ./ moment_0
  #moment_2 = vcat(sol.u'...)[:, 3]
  
  print("t_cloudy=",time,"\n")
  print("M0_cloudy=",moment_0,"\n")
  print("M1_cloudy=",moment_1,"\n")
  #print("M2_cloudy=",moment_2,"\n")

  # track the precipitable mass
  t_precip = collect(range(tspan[1], stop=tspan[2], length=100))
  m_precip = zeros(FT, length(t_precip))
  for (i,t) in enumerate(t_precip)
    mj_t = sol(t)
    M0 = mj_t[1]
    M1 = mj_t[2]
    n = M0
    k = 1.0
    θ = M1 / M0
    psd = x -> n ./ θ .* exp.(-x ./ θ)
    integrand = x-> x * psd(x)
    m_precip[i] = quadgk(integrand, v_cutoff, vmax)[1]
  end
  println("t_precip = ",t_precip)
  println("v_precip = ",m_precip)

  # output results to file
  m_end = sol(tspan[end])
  M0 = m_end[1]
  M1 = m_end[2]
  n = M0
  k = 1.0
  θ = M1 / M0
  M2 = 2*n*θ.^2
  open(string(casename,"results.txt"),"a") do file
    write(file,string("final N, k, theta = ", n,", ", k,", ", θ,", ","\n"))
    write(file,string("t_cloudy = ", time,"\n"))
    write(file,string("M0_cloudy = ", moment_0,"\n"))
    write(file,string("M1_cloudy = ", moment_1,"\n"))
    write(file,string("M2_cloudy = ", moment_2,"\n"))
    write(file,string("t_precip = ", t_precip,"\n"))
    write(file,string("m_precip = ", m_precip,"\n"))
  end

end

main()
