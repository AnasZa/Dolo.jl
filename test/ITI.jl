
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
compute_radius=false
verbose=true
maxit =1000
verbose_jac=false

parms = model.calibration[:parameters]

n_m = Dolo.n_nodes(dprocess)
n_mt = Dolo.n_inodes(dprocess,1)
n_s = length(model.symbols[:states])

s = Dolo.nodes(model.grid)
N_s = size(s,1)
n_x = size(model.calibration[:controls],1)
N_m = Dolo.n_nodes(dprocess) # number of grid points for exo_vars

#  x0 = [repmat(model.calibration[:controls]',N_s) for i in 1:N_m] #n_x N_s n_m
dprocess = Dolo.discretize( model.exogenous )
init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
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

verbose && println("N\tf_x\t\td_x\tTime_residuals\tTime_inversion\tTime_search\tLambda_0\tN_invert\tN_search\t")
verbose && println(repeat("-", 120))


if compute_radius == true
  res=zeros(res_init)
  dres = zeros(N_s*N_m, n_x, n_x)
end



it += 1

jres = zeros(n_m,n_mt,N_s,n_x,n_x)
S_ij = zeros(n_m,n_mt,N_s,n_s)

t1 = time();

# compute derivatives and residuals:
# res: residuals
# dres: derivatives w.r.t. x
# jres: derivatives w.r.t. ~x
# fut_S: future states
Dolo.set_values!(ddr,x)

ff = SerialDifferentiableFunction(u-> euler_residuals(model, s, u,ddr,dprocess,parms;
                                  with_jres=false,set_dr=false))

res, dres = ff(x)

# dres = permutedims(dres, [axisdim(dres, Axis{:n_v}),axisdim(dres, Axis{:N}),axisdim(dres, Axis{:n_x})])
dres = reshape(dres, n_m, N_s, n_x, n_x)
junk, jres, fut_S = euler_residuals(model, s, x,ddr,dprocess,parms, with_jres=true,set_dr=false, jres=jres, S_ij=S_ij)
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
t2 = time();
tot, it_invert, lam0 = invert_jac(res,dres,jres,fut_S, ddr_filt; verbose=verbose_jac, maxit = 10)

t3 = time();

i_bckstps=0
new_err=err_0
new_x = x
while new_err>=err_0 && i_bckstps<length(steps)
  i_bckstps +=1
  new_x = x-destack0(tot, n_m)*steps[i_bckstps]
  new_res = euler_residuals(model, s, new_x,ddr,dprocess,parms,set_dr=true)
  new_err = maximum(abs, new_res)
end
err_2 = maximum(abs,tot)

t4 = time();

x = new_x
verbose && @printf "%-6i% -10e% -17e% -15.4f% -15.4f% -15.5f% -17.3f%-17i%-5i\n" it  err_0  err_2  t2-t1 t3-t2 t4-t3 lam0 it_invert i_bckstps




















while it <= maxit && err_0>tol
   it += 1

   jres = zeros(n_m,n_m,N_s,n_x,n_x)
   S_ij = zeros(n_m,n_m,N_s,n_s)

   t1 = time();

   # compute derivatives and residuals:
   # res: residuals
   # dres: derivatives w.r.t. x
   # jres: derivatives w.r.t. ~x
   # fut_S: future states
   Dolo.set_values!(ddr,x)

   ff = SerialDifferentiableFunction(u-> euler_residuals(model, s, u,ddr,dprocess,parms;
                                     with_jres=false,set_dr=false))

   res, dres = ff(x)

   # dres = permutedims(dres, [axisdim(dres, Axis{:n_v}),axisdim(dres, Axis{:N}),axisdim(dres, Axis{:n_x})])
   dres = reshape(dres, n_m, N_s, n_x, n_x)
   junk, jres, fut_S = euler_residuals(model, s, x,ddr,dprocess,parms, with_jres=true,set_dr=false, jres=jres, S_ij=S_ij)
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
   t2 = time();
   tot, it_invert, lam0 = invert_jac(res,dres,jres,fut_S, ddr_filt; verbose=verbose_jac, maxit = smaxit)

   t3 = time();

   i_bckstps=0
   new_err=err_0
   new_x = x
   while new_err>=err_0 && i_bckstps<length(steps)
     i_bckstps +=1
     new_x = x-destack0(tot, n_m)*steps[i_bckstps]
     new_res = euler_residuals(model, s, new_x,ddr,dprocess,parms,set_dr=true)
     new_err = maximum(abs, new_res)
   end
   err_2 = maximum(abs,tot)

   t4 = time();

   x = new_x
   verbose && @printf "%-6i% -10e% -17e% -15.4f% -15.4f% -15.5f% -17.3f%-17i%-5i\n" it  err_0  err_2  t2-t1 t3-t2 t4-t3 lam0 it_invert i_bckstps

end
Dolo.set_values!(ddr,x)























tot, it_invert, lam0 = invert_jac(res,dres,jres,fut_S, ddr_filt; verbose=verbose_jac)
tot
t3 = time();

i_bckstps=0
new_err=err_0
new_x = x

i_bckstps +=1
new_x = x-destack0(tot, n_m)*steps[i_bckstps]
new_res = euler_residuals(model,s,new_x,ddr,dprocess,parms,set_dr=true)
new_err = maximum(abs, new_res)




while new_err>=err_0 && i_bckstps<length(steps)
  i_bckstps +=1
  new_x = x-destack0(tot, n_m)*steps[i_bckstps]
  new_res = euler_residuals(model,s,new_x,ddr,dprocess,parms,set_dr=true)
  new_err = maximum(abs, new_res)
end
err_2 = maximum(abs,tot)

t4 = time();

x = new_x



println("N\tf_x\t\td_x\tTime_residuals\tTime_inversion\tTime_search\tLambda_0\tN_invert\tN_search\t")

println(repeat("-", 120))
# println(it, "\t", err_0, "\t", err_2, "\t", t2-t1, "\t", t3-t2, "\t", t4-t3, "\t", lam0, "\t", it_invert, "\t", i_bckstps)



# @printf "%-6i% -10e% -10e% -10e% -10e% -10e% -10e%-10e%-5i\n" it  err_0  err_2  round(t2-t1,4) round(t3-t2,4) round(t4-t3,4) lam0 it_invert i_bckstps
@printf "%-6i% -10e% -15e% -15.4f% -15.4f% -15.5f% -17i%-17i%-5i\n" it  err_0  err_2  t2-t1 t3-t2 t4-t3 lam0 it_invert i_bckstps


round(t2-t1,2)

int width1, width2;
int values[6][2];
printf("|%s%n|%s%n|\n", header1, &width1, header2, &width2);

for(i=0; i<6; i++)
   printf("|%*d|%*d|\n", width1, values[i][0], width2, values[i][1]);

Info=ITIDetails(err_0, err_2,  t2-t1, t3-t2, t4-t3, lam0, it_invert, i_bckstps)


Info.f_x

println(Info)





@printf "%.2f"  err_0

println([err_0])
println(err_0)
@printf string(err_0)

string(err_0)
print_shortest(err_0)
showall(err_0);println()
fname = "simple.dat"
# using do means the file is closed automatically
# in the same way "with" does in python
open(fname,"r") do f
    for line in eachline(f)
        print(line)
    end
end


ptable = DataFrame( @data([1,   2,    6,    8,    26    ]))
println(ptable)
function Iterationslogger(x::ITIDetails)








95000/0.05

println(repeat("-", 115))
map(Info -> println(
    Info.f_x,                   "\t",
    Info.d_x,   "\t",
    Info.Time_residuals, "\t",
    Info.Time_inversion,    "\t",
    Info.Time_search,   "\t",
    Info.Lambda_0,    "\t",
    Info.N_invert,     "\t",
    Info.N_search),
    Info);




ITIDetails(err_0, err_2, t2-t1, t3-t2, t4-t3, lam0, it_invert, i_bckstps)

converged(r::ITI_headers) = r.header
function Base.print(io::IO, r::ITIDetails)
  @printf io " * Number of iterations: %s\n" string(r.f_x)
  @printf io " * Decision Rule type: %s\n" string(r.d_x)
  @printf io " * Convergence: %s\n" string(r.Time_residuals)
  @printf io " * Convergence: %s\n" string(r.Time_inversion)
  @printf io " * Convergence: %s\n" string(r.Time_search)
  @printf io " * Convergence: %s\n" string(r.Lambda_0)
  @printf io " * Convergence: %s\n" string(r.N_invert)
  @printf io " * Convergence: %s\n" string(r.N_search)
end

ITIDetails(err_0, err_2, t2-t1, t3-t2, t4-t3, lam0, it_invert, i_bckstps)





println(ITIDetails)

DataFrame(Info)



using DataFrames
DataFrames.DataFrame(Info)














function Base.show(io::IO, r::ITI_headers)
    @printf io "Results of Improved Time Iteration Algorithm\n"
    @printf io " * Number of iterations: %s\n" string(r.f_x)
    # @printf io " * Complementarities: %s\n" string(r.complementarities)
    @printf io " * Decision Rule type: %s\n" string(r.d_x)
    @printf io " * Convergence: %s\n" string(r.Time_residuals)
    @printf io " * Contractivity: %s\n" string(r.Time_residuals)
    @printf io "   * |x - x'| < %.1e: %s\n" string(r.Time_residuals)
    @printf io "   * |x - x'| < %.1e: %s\n" string(r.Time_inversion)
    @printf io "   * |x - x'| < %.1e: %s\n" string(r.Time_search)
    @printf io "   * |x - x'| < %.1e: %s\n" string(r.Lambda_0)
    @printf io "   * |x - x'| < %.1e: %s\n" string(r.N_invert)
    @printf io "   * |x - x'| < %.1e: %s\n" string(r.N_search)
end



type ITIDetails
    f_x::Float64
    d_x::Float64
    Time_residuals::Float64
    Time_inversion::Float64
    Time_search::Float64
    Lambda_0::Float64
    N_invert::Float64
    N_search::Float64
end


println("N\tf_x\td_x\tTime_residuals\tTime_inversion\tTime_search\tLambda_0\tN_invert\tN_search\t")
println(repeat("-", 115))
map((xcol,ycol) -> println(
    xcol,                   "\t",
    mean(anscombe[xcol]),   "\t",
    median(anscombe[xcol]), "\t",
    std(anscombe[xcol]),    "\t",
    mean(anscombe[ycol]),   "\t",
    std(anscombe[ycol]),    "\t",
    cor(anscombe[xcol], anscombe[ycol]))


x
Dolo.set_values!(ddr,x)

ImprovedTimeIterationResult(ddr.dr, it, err_0, err_2, tol, lam0, it_invert, 5.0)
ImprovedTimeIterationResult(ddr.dr, it)


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
