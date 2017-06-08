
path = Pkg.dir("Dolo")

# Pkg.build("QuantEcon")
import Dolo
import Bruteforce_module

###############################################################################
filename = joinpath(path,"examples","models","rbc_dtcc_mc.yaml")
# model = Dolo.Model(Pkg.dir("Dolo", "examples", "models", "rbc_dtcc_mc.yaml"), print_code=true)
model = Dolo.yaml_import(filename)

@time dr_ITI  = Bruteforce_module.improved_time_iteration(model; verbose=true, tol = 1e-06, smaxit=50)


# dprocess = Dolo.discretize( model.exogenous )
# init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
# @time dr_ITI_2  = Bruteforce_module.improved_time_iteration(model, dprocess,init_dr)
#
@time dr_TI  = Dolo.time_iteration(model;tol_η=1e-08, maxit=1000)


################################################################################
function profile_ITI(m)
    dr_ITI  = Bruteforce_module.improved_time_iteration(m)
    return dr_ITI
end


using ProfileView

profile_ITI(model)
Profile.clear()
@profile profile_ITI(model)
ProfileView.view()




###############################################################################
#Plotting the DR

# ITIT method
df_ITI = Dolo.tabulate(model, dr_ITI.dr, :k)

import PyPlot
plt = PyPlot;
fig = plt.figure("Decision Rule, ITI-method",figsize=(8.5,5))

plt.subplot(1,2,1)
plt.plot(df_ITI[:k], df_ITI[:n])
plt.ylabel("Hours");
plt.xlabel("state = k");
plt.title("Decision Rule");

plt.subplots_adjust(wspace=1)
# plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(df_ITI[:k], df_ITI[:i])
plt.ylabel("Investment");
plt.xlabel("state = k");
plt.title("Decision Rule");
###############################################
# TI method
df = Dolo.tabulate(model, dr_TI.dr, :k)


fig = plt.figure("Decision Rule, IT-method",figsize=(8.5,5))

plt.subplot(1,2,1)
plt.plot(df[:k], df[:n])
plt.ylabel("Hours");
plt.xlabel("state = k");
plt.title("Decision Rule");

plt.subplots_adjust(wspace=1)
# plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(df[:k], df[:i])
plt.ylabel("Investment");
plt.xlabel("state = k");
plt.title("Decision Rule");

################################################################################
# Try for other models

filename = joinpath(path,"examples","models","rbc_dtcc_ar1.yaml")
model_ar1 = Dolo.yaml_import(filename)
@time dr_ITI_ar1  = Bruteforce_module.improved_time_iteration(model_ar1; verbose=true, tol = 1e-06, smaxit=50)



filename = joinpath(path,"examples","models","rbc_dtcc_mc.yaml")
# model = Dolo.Model(Pkg.dir("Dolo", "examples", "models", "rbc_dtcc_mc.yaml"), print_code=true)
model = Dolo.yaml_import(filename)


model.grid.nodes


dprocess = Dolo.discretize( model.exogenous )
grid=Dict()
grid= Dolo.get_grid(model, options=grid)
endo_nodes = Dolo.nodes(grid)
size(endo_nodes, 2)
Dolo.n_nodes(dprocess)
N_m=1
P = [Dolo.node(dprocess, i) for i in 1:N_m]

[repmat(Dolo.node(dprocess,1)',1) for i in 1:50]

Dolo.node(dprocess,1)'

m = Dolo.node(dprocess, 1)
Dolo.inode(dprocess, 2, 1)
Dolo.n_inodes(dprocess, 1)   # N_m
Dolo.iweight(dprocess, 1, 1)

P = dprocess.values
m_prep = [repmat(P[1,:]',1) for i in 1:50]
m=(hcat([e' for e in m_prep]...))'

for i in 1:size(res, 1)
    m = node(dprocess, i)
    for j in 1:n_inodes(dprocess, i)
        M = inode(dprocess, i, j)
        w = iweight(dprocess, i, j)
        # Update the states
        for n in 1:N
            S[n, :] = Dolo.transition(model, m, s[n, :], x[i][n, :], M, p)
        end

        X = dr(i, j, S)
        for n in 1:N
            res[i][n, :] += w*Dolo.arbitrage(model, m, s[n, :], x[i][n, :], M, S[n, :], X[n, :], p)
        end
    end
end






init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
@time dr_ITI_ar1  = Bruteforce_module.improved_time_iteration(model_ar1, dprocess, init_dr; verbose=true, tol = 1e-06, smaxit=50)

model =model_ar1
parms = model.calibration[:parameters]
# need to discretize if continous MC
# dprocess = model.exogenous

nodes = dprocess.values
transitions = dprocess.transitions
n_m = size(nodes,1)
n_s = length(model.symbols[:states])
Dolo.node(dprocess, 2)

Dolo.n_nodes(dprocess)
