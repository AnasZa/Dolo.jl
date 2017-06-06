"""
Computes a global solution for a model via backward Improved Time Iteration. The algorithm is applied to the residuals of the arbitrage equations. The idea is to solve the system G(x) = 0 as a big nonlinear system in x, where the inverted Jacobian matrix is approximated by an infinite sum (Neumann series).

If the initial guess for the decision rule is not explicitly provided, the initial guess is provided by `ConstantDecisionRule`.
If the stochastic process for the model is not explicitly provided, the process is taken from the default provided by the model object, `model.exogenous`

# Arguments
* `model::NumericModel`: Model object that describes the current model environment.
* `dprocess`: The stochastic process associated with the exogenous variables in the model.
* `init_dr`: Initial guess for the decision rule.
* `maxbsteps` Maximum number of backsteps.
* `verbose` Set "true" if you would like to see the details of the infinite sum convergence.
* `smaxit` Maximum number of iterations to compute the Neumann series.
* `complementarities`
* `compute_radius`
* `details` If false returns only a decision rule dr
# Returns
* `dr`: Solved decision rule.
* `details` about the iterations is specified.
"""

function improved_time_iteration(model:: Dolo.AbstractModel, dprocess::Dolo.AbstractDiscretizedProcess,
                                 init_dr::Dolo.AbstractDecisionRule;
                                 maxbsteps::Int=10, verbose::Bool=false,
                                 tol::Float64=1e-8, smaxit::Int=500, maxit::Int=1000,
                                 complementarities::Bool=false, compute_radius::Bool=false, details::Bool=true)

  #  f = Dolo.arbitrage
  #  g = Dolo.transition
   # x_lb = model.functions['controls_lb']
   # x_ub = model.functions['controls_ub']

   parms = model.calibration[:parameters]
   # need to discretize if continous MC
   dprocess = model.exogenous

   nodes = dprocess.values
   transitions = dprocess.transitions
   n_m = size(nodes,1)
   n_s = length(model.symbols[:states])

   # endo grid today
   s = model.grid.nodes
   # controls today
   N_s = size(s,1)
   n_x = size(model.calibration[:controls],1)
   N_m = Dolo.n_nodes(dprocess.grid) # number of grid points for exo_vars

  #  x0 = [repmat(model.calibration[:controls]',N_s) for i in 1:N_m] #n_x N_s n_m
   x0 = [init_dr(i, Dolo.nodes(model.grid)) for i=1:N_m]
   ddr=Dolo.CachedDecisionRule(dprocess, model.grid, x0)
   ddr_filt = Dolo.CachedDecisionRule(dprocess, model.grid, x0)
   Dolo.set_values!(ddr,x0)

   steps = 0.5.^collect(0:maxbsteps)

   x=x0
   ## memory allocation
   jres = zeros(n_m,n_m,N_s,n_x,n_x)
   S_ij = zeros(n_m,n_m,N_s,n_s)

   ######### Loop     for it in range(maxit):
   it=0
   it_invert=0
   res_init = euler_residuals(model,s,x,ddr,dprocess,parms ,set_dr=false, jres=jres, S_ij=S_ij)
   err_0 = abs(maximum(res_init))
   err_2= err_0
   lam0=0.0

   if compute_radius == true
     res=zeros(res_init)
     dres = zeros(N_s*N_m, n_x, n_x)
   end

   while it <= maxit && err_0>tol
      it += 1

      jres = zeros(n_m,n_m,N_s,n_x,n_x)
      S_ij = zeros(n_m,n_m,N_s,n_s)

      # compute derivatives and residuals:
      # res: residuals
      # dres: derivatives w.r.t. x
      # jres: derivatives w.r.t. ~x
      # fut_S: future states
      Dolo.set_values!(ddr,x)

      ff = SerialDifferentiableFunction(u-> euler_residuals(model,s,u,ddr,dprocess,parms;
                                        with_jres=false,set_dr=false))

      res, dres = ff(x)

      # dres = permutedims(dres, [axisdim(dres, Axis{:n_v}),axisdim(dres, Axis{:N}),axisdim(dres, Axis{:n_x})])
      dres = reshape(dres, 2,50,2,2)
      junk, jres, fut_S = euler_residuals(model,s,x,ddr,dprocess,parms, with_jres=true,set_dr=false, jres=jres, S_ij=S_ij)
        # if there are complementerities, we modify derivatives
      err_0 = abs(maximum(res))

      jres *= -1.0
      jres[1,1,1:5,:,:]
      M=jres
      # M[1,1,1:5,:,:]

      X=zeros(n_m,N_s,n_x,n_x)
      for i_m in 1:n_m
          for j_m in 1:n_m
              # M = jres[i_m,j_m,:,:,:]
              X = deepcopy(dres[i_m,:,:,:])
              for n in 1:N_s
                 X[n,:,:], M[i_m,j_m,n,:,:] = invert(collect(X[n,:,:]), M[i_m,j_m,n,:,:])
              end
          end
      end

      ####################
      # Invert Jacobians

      tot, it_invert, lam0 = invert_jac(res,dres,jres,fut_S, ddr_filt; verbose=verbose)

      i_bckstps=0
      new_err=err_0
      new_x = x
      while new_err>=err_0 && i_bckstps<length(steps)
        i_bckstps +=1
        new_x = x-destack0(tot, n_m)*steps[i_bckstps]
        new_res = euler_residuals(model,s,new_x,ddr,dprocess,parms,set_dr=true)
        new_err = maximum(abs, new_res)
      end
      err_2 = maximum(abs,tot)
      x = new_x
   end
   Dolo.set_values!(ddr,x)
   lam, lam_max, lambdas = radius_jac(res,dres,jres,S_ij,ddr_filt)

   if !details
     return ddr.dr
   else
     converged = err_0<tol
     return ImprovedTimeIterationResult(ddr.dr, it, err_0, err_2, converged, tol, lam0, it_invert, 5.0), (lam, lam_max, lambdas)
   end

end

function improved_time_iteration(model, dprocess::Dolo.AbstractDiscretizedProcess, maxbsteps::Int=10, verbose::Bool=false,
                                 tol::Float64=1e-8, smaxit::Int=500, maxit::Int=1000,
                                 complementarities::Bool=true, compute_radius::Bool=false)
    init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
    return improved_time_iteration(model, dprocess, init_dr, maxbsteps, verbose,tol,
                                    smaxit, maxit,complementarities, compute_radius)
end

# function improved_time_iteration(model, maxbsteps::Int=10, verbose::Bool=false,
#                                  tol::Float64=1e-8, smaxit::Int=500, maxit::Int=1000,
#                                  complementarities::Bool=true, compute_radius::Bool=false)
#     dprocess = Dolo.discretize( model.exogenous )
#     init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
#     return improved_time_iteration(model, dprocess, init_dr, maxbsteps, verbose,tol,
#                                    smaxit, maxit,complementarities, compute_radius)
# end

function improved_time_iteration(model; kwargs...)
    dprocess = Dolo.discretize( model.exogenous )
    init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
    return improved_time_iteration(model, dprocess, init_dr; kwargs...)
end
