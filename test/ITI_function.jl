function improved_time_iteration(model:: Dolo.AbstractModel, dprocess,
                                 init_dr::Dolo.AbstractDecisionRule;
                                 maxbsteps::Int=10, verbose::Bool=false,
                                 tol::Float64=1e-8, smaxit::Int=500, maxit::Int=1000,
                                 complementarities::Bool=false, compute_radius::Bool=false)

   f = Dolo.arbitrage
   g = Dolo.transition
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
  #  ddr=Dolo.DecisionRule(dprocess.grid, model.grid, n_x)
   ddr=Dolo.CachedDecisionRule(dprocess, model.grid, x0)
  #  ddr_filt = Dolo.DecisionRule(dprocess.grid, model.grid,n_x)
   ddr_filt = Dolo.CachedDecisionRule(dprocess, model.grid, x0)
  #  ddr== ddr_filt
   Dolo.set_values!(ddr,x0)

   steps = 0.5.^collect(0:maxbsteps)

   x=x0
   ## memory allocation
   jres = zeros(n_m,n_m,N_s,n_x,n_x)
   S_ij = zeros(n_m,n_m,N_s,n_s)

   ######### Loop     for it in range(maxit):
   it=0
   it_invert=0
   res_init = euler_residuals(f,g,s,x,ddr,dprocess,parms ,set_dr=false, jres=jres, S_ij=S_ij)
   err_0 = abs(maximum(res_init))
   err_2= err_0
   lam0=0.0
   while it <= maxit && err_0>tol
      it += 1
     #  println(it)
      jres = zeros(n_m,n_m,N_s,n_x,n_x)
      S_ij = zeros(n_m,n_m,N_s,n_s)

      # compute derivatives and residuals:
      # res: residuals
      # dres: derivatives w.r.t. x
      # jres: derivatives w.r.t. ~x
      # dres: derivatives w.r.t. x
      # jres: derivatives w.r.t. ~x
      # fut_S: future states
      Dolo.set_values!(ddr,x)

      ff = SerialDifferentiableFunction(u-> euler_residuals(f,g,s,u,ddr,dprocess,parms;
                                        with_jres=false,set_dr=false))

      res, dres = ff(x)

      # dres = permutedims(dres, [axisdim(dres, Axis{:n_v}),axisdim(dres, Axis{:N}),axisdim(dres, Axis{:n_x})])
      dres = reshape(dres, 2,50,2,2)

      junk, jres, fut_S = euler_residuals(f,g,s,x,ddr,dprocess,parms, with_jres=true,set_dr=false, jres=jres, S_ij=S_ij)
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

      tot, it_invert, lam0 = invert_jac(res,dres,jres,fut_S; verbose=true,filt=ddr_filt)

      i_bckstps=0
      new_err=err_0
      new_x = x
      while new_err>=err_0 && i_bckstps<length(steps)
        i_bckstps +=1
        new_x = x-destack0(tot, n_m)*steps[i_bckstps]
        new_res = euler_residuals(f,g,s,new_x,ddr,dprocess,parms,set_dr=true)
        new_err = maximum(abs, new_res)
      end
      err_2 = maximum(abs,tot)
      x = new_x
   end
   Dolo.set_values!(ddr,x)
  #  ImprovedTimeIterationResult(ddr, it, err_0, err_2, tol, lam0, it_invert, 5.0),
  if verbose ==true
    return  ImprovedTimeIterationResult(ddr.dr, it, err_0, err_2, tol, lam0, it_invert, 5.0)
  else
    return ddr.dr
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
