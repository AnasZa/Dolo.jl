
path = Pkg.dir("Dolo")

# Pkg.build("QuantEcon")
import Dolo
using AxisArrays
include("bruteforce_help.jl")
include("ITI_function.jl")

###############################################################################
# filename = joinpath(path,"examples","models","rbc_dtcc_mc.yaml")
# # model = Dolo.Model(Pkg.dir("Dolo", "examples", "models", "rbc_dtcc_mc.yaml"), print_code=true)
# model = Dolo.yaml_import(filename)
#
#
# @time dr_ITI  = improved_time_iteration(model)
#
# dprocess = Dolo.discretize( model.exogenous )
# init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
# @time dr_ITI_2  = improved_time_iteration(model, dprocess,init_dr)
#
# @time dr_TI  = Dolo.time_iteration(model)
#
# typeof(Array{Float64,3})<:AbstractVector


##################################################################
filename = joinpath(path,"examples","models","rbc_dtcc_mc.yaml")
# model = Dolo.Model(Pkg.dir("Dolo", "examples", "models", "rbc_dtcc_mc.yaml"), print_code=true)
model = Dolo.yaml_import(filename)
s = model.grid.nodes
# controls today
N_s = size(s,1)
n_x = size(model.calibration[:controls],1)
N_m = Dolo.n_nodes(model.exogenous.grid)

maxbsteps=10
tol=1e-08
typeof(maxbsteps)

# f = Dolo.arbitrage
# g = Dolo.transition
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


# x0 = repmat([(model.calibration[:controls])'],N*n_x^2,1)
# x0 = repmat(model.calibration[:controls]',N)
# x0 = [repmat(model.calibration[:controls]',N_s) for i in 1:N_m] #n_x N_s n_m

init_dr=Dolo.ConstantDecisionRule(model.calibration[:controls])

x0 = [init_dr(i, Dolo.nodes(model.grid)) for i=1:N_m]

# ddr=Dolo.DecisionRule(dprocess.grid, model.grid, n_x)
# ddr_filt = Dolo.DecisionRule(dprocess.grid, model.grid,n_x)

ddr = Dolo.CachedDecisionRule(dprocess, model.grid, x0)
ddr_filt = Dolo.CachedDecisionRule(dprocess, model.grid, x0)

typeof(ddr.dr)<:Dolo.AbstractDecisionRule

ddr== ddr_filt
Dolo.set_values!(ddr,x0)

steps = 0.5.^collect(0:maxbsteps)

x=x0


# checking the euler_residuals functions, res = 0 ##########################
# @time dr = Dolo.time_iteration(model, verbose=true, maxit=10000, details=false)
# euler_residuals(f,g,s,x,dr,dprocess,parms, with_jres=false,set_dr=true)
# Doesn't seem to work, but the same thing in python ...

## memory allocation
jres = zeros(n_m,n_m,N_s,n_x,n_x)
S_ij = zeros(n_m,n_m,N_s,n_s)

# fut_S = copy(S_ij)
######### Loop     for it in range(maxit):
it=0
it_invert=0
tol=1e-08
maxit=1000
# err_0=100
# err_2=100

res_init = euler_residuals(model,s,x,ddr,dprocess,parms ,set_dr=false, jres=jres, S_ij=S_ij)
raduis_jac=true
if raduis_jac == true
  res=zeros(res_init)
  dres = zeros(N_s*N_m, n_x, n_x)
end
  # if there are complementerities, we modify derivatives
err_0 = maximum(abs, res_init)
err_2= 0.0
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

   ff = SerialDifferentiableFunction(u-> euler_residuals(model,s,u,ddr,dprocess,parms;
                                     with_jres=false,set_dr=false))

   res, dres = ff(x)

   # dres = permutedims(dres, [axisdim(dres, Axis{:n_v}),axisdim(dres, Axis{:N}),axisdim(dres, Axis{:n_x})])
   dres = reshape(dres, 2,50,2,2)

  #  junk, jres, fut_S = euler_residuals(model,s,x,ddr,dprocess,parms, with_jres=true,set_dr=false, jres=jres, S_ij=S_ij)
   junk, jres, S_ij = euler_residuals(model,s,x,ddr,dprocess,parms, with_jres=true,set_dr=false, jres=jres, S_ij=S_ij)
  #  S_ij=fut_S
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

  #  tot, it_invert, lam0 = invert_jac(res,dres,jres,fut_S,ddr_filt; verbose=true)
   tot, it_invert, lam0 = invert_jac(res,dres,jres,S_ij,ddr_filt; verbose=true)

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

   # print...
   x = new_x
   println(it)
end
x
Dolo.set_values!(ddr,x)

ImprovedTimeIterationResult(ddr.dr, it, err_0, err_2, tol, lam0, it_invert, 5.0)


###### radius_jac
function radius_jac(res::AbstractArray,dres::AbstractArray,jres::AbstractArray,S_ij::AbstractArray,
                    ddr_filt; tol=tol, maxit=1000, verbose=false, precomputed = false)

   n_m, N_s, n_x = size(res)
   err0 = 0.0
   ddx = rand(size(res))*10^9

   lam = 0.0
   lam_max = 0.0

   lambdas = zeros(maxit)
   if verbose==true
       print("Starting inversion. Radius_Jac")
   end

   for nn in 1:maxit
     ddx /= maximum(abs, ddx)
     d_filt_dx(ddx,jres,S_ij,dumdr; precomputed=precomputed)
     lam = maximum(abs, ddx)
     lam_max = max(lam_max, lam)
     lambdas[nn] = lam
   end

   return (lam, lam_max, lambdas)

 end

dres
lam, lam_max, lambdas = radius_jac(res,dres,jres,S_ij,ddr_filt)

typeof(S_ij)<:AbstractArray


n_m, N_s, n_x = size(res_init)

err0 = 0.0

# import numpy.random
# rand(size(res_init))
ddx = rand(size(res_init))*10000000000

# if filt == nothing
#   error("No filter supplied.")
# else
#   dumdr = filt
# end
dumdr = ddr_filt

# if isinstance(dumdr, SmolyakDecisionRule):
#     for i in range(n_m):
#         for j in range(n_m):
#             dumdr.precompute_Phi(i,j,fut_S[i,j,...])
#     precomputed = True
# else:
    # precomputed = False
precomputed = false
# jres[...] *= -1.0
# jres = -jres

lam = 0.0
lam_max = 0.0

lambdas = zeros(maxit)
if verbose==true
    print("Starting inversion. Radius_Jac")
end

for nn in 1:maxit
  ddx /= maximum(abs, ddx)
  d_filt_dx(ddx,jres,S_ij,dumdr; precomputed=precomputed)
  lam = maximum(abs, ddx)
  lam_max = max(lam_max, lam)
  lambdas[nn] = lam
end

lambdas
# return (lam, lam_max, lambdas)
