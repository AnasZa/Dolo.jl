
path = Pkg.dir("Dolo")

# Pkg.build("QuantEcon")
import Dolo
using AxisArrays
include("bruteforce_help.jl")


###############################################################################

filename = joinpath(path,"examples","models","rbc_dtcc_mc.yaml")
# model = Dolo.Model(Pkg.dir("Dolo", "examples", "models", "rbc_dtcc_mc.yaml"), print_code=true)
model = Dolo.yaml_import(filename)

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


# x0 = repmat([(model.calibration[:controls])'],N*n_x^2,1)
# x0 = repmat(model.calibration[:controls]',N)
x0 = [repmat(model.calibration[:controls]',N_s) for i in 1:N_m] #n_x N_s n_m
ddr=Dolo.DecisionRule(dprocess.grid, model.grid, n_x)
ddr_filt = Dolo.DecisionRule(dprocess.grid, model.grid,n_x)
ddr== ddr_filt
Dolo.set_values!(ddr,x0)


x=x0


# checking the euler_residuals functions, res = 0 ##########################
@time dr = Dolo.time_iteration(model, verbose=true, maxit=10000, details=false)
euler_residuals(f,g,s,x,dr,dprocess,parms, with_jres=false,set_dr=true)
# Doesn't seem to work, but the same thing in python ...

## memory allocation
jres = zeros(n_m,n_m,N_s,n_x,n_x)
S_ij = zeros(n_m,n_m,N_s,n_s)

######### Loop     for it in range(maxit):

it = 1

# compute derivatives and residuals:
# res: residuals
# dres: derivatives w.r.t. x
# jres: derivatives w.r.t. ~x
# dres: derivatives w.r.t. x
# jres: derivatives w.r.t. ~x
# fut_S: future states
# ddr.set_values(x)  again????

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

tot, it, lam0 = invert_jac(res,dres,jres,fut_S; verbose=true,filt=ddr_filt)
it
lam0
# function invert_jac(res,dres,jres,fut_S; filt= nothing, tol=1e-10, maxit=1000, verbose=false)
tot[:,3,:]
