
path = Pkg.dir("Dolo")

# Pkg.build("QuantEcon")
import Dolo

###################################

# Why AbstractDecisionRule is not defined?
function euler_residuals(f, g, s::AbstractArray, x::AbstractArray, dr,
                         dprocess, parms::AbstractArray; with_jres=false, set_dr=true,
                         jres=nothing, S_ij=nothing)

    if set_dr ==trues
      Dolo.set_values!(dr,x)
    end

    N_s = size(s,1)
    n_s = size(s,2)
    n_x = size(x,1)

    P = dprocess.values
    Q = dprocess.transitions

    n_ms = size(P,1)  # number of markov states
    n_mv = size(P,2)  # number of markov variable

    res = zeros(n_ms, N_s, n_x)

    if with_jres == true
      if jres== nothing
        jres = zeros((n_ms,n_ms,N_s,n_x,n_x))
      end
      if jreS_ijs== nothing
        S_ij = zeros((n_ms,n_ms,N_s,n_s))
      end
    end


    for i_ms in 1:n_ms
       m_prep = [repmat(P[i_ms,:]',1) for i in 1:N_s]
       m=(hcat([e' for e in m_prep]...))'
       xm = x[i_ms]

       for I_ms in 1:n_ms
          M_prep = [repmat(P[I_ms,:]', 1) for i in 1:N_s]
          M=(hcat([e' for e in M_prep]...))'
          prob = Q[i_ms, I_ms]

          S = g(model, m, s, xm, M, parms)
          XM = dr(I_ms, S)

          if with_jres==true
              ff = SerialDifferentiableFunction(u->f(model, m,s,xm,M,S,u,parms))
              rr, rr_XM = ff(XM)
              jres[i_ms,I_ms,:,:,:] = prob*rr_XM
              S_ij[i_ms,I_ms,:,:] = S
          else
              rr = f(model, m,s,xm,M,S,XM,parms)
              res[i_ms,:,:] += prob*rr
          end

        end

    end

    if with_jres==true
        return res, jres, S_ij
    else
        return res
    end

end



######################
function SerialDifferentiableFunction(f, epsilon=1e-8)

    function df(x)

        v0 = f(x)

        N = size(v0,1)
        n_v = size(v0,2)
        assert(size(x,1) == N)
        n_x = size(x,1)

        dv = zeros((N,  n_v, n_x, N))

        for i in 1:n_x
          # not sure we need to copy(x)
          xi = x
          x[i_ms]
          [xi[j][:,i] += epsilon for j in 1:N]
          vi = f(xi)
          dv[:,:,:, i] = (vi+(-1*v0))./epsilon
        end

        return [v0, dv]
    end
end

#############################################################################


filename = joinpath(path,"examples","models","rbc_dtcc_mc.yaml")
# model = Dolo.Model(Pkg.dir("Dolo", "examples", "models", "rbc_dtcc_mc.yaml"), print_code=true)
model = Dolo.yaml_import(filename)
@time dr = Dolo.time_iteration(model, verbose=true, maxit=10000, details=false)



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
N_m = Dolo.n_nodes(dprocess.grid)


# x0 = repmat([(model.calibration[:controls])'],N*n_x^2,1)
# x0 = repmat(model.calibration[:controls]',N)
x0 = [repmat(model.calibration[:controls]',N_s) for i in 1:N_m] #n_x N_s N_m
ddr=Dolo.DecisionRule(dprocess.grid,model.grid, n_x)

Dolo.set_values!(ddr,x0)

x=x0

## memory allocation
jres = zeros(n_m,n_m,N_s,n_x,n_x)
S_ij = zeros(n_m,n_m,N_s,n_s)

######### Loop     for it in range(maxit):

it = 1

# compute derivatives and residuals:
# res: residuals
# dres: derivatives w.r.t. x
# jres: derivatives w.r.t. ~x
# fut_S: future states
# ddr.set_values(x)  again????

ff = SerialDifferentiableFunction(u-> euler_residuals(f,g,s,u,ddr,dprocess,parms;
                                  with_jres=false,set_dr=false))


euler_residuals(f,g,s,x,ddr,dprocess,parms; with_jres=false,set_dr=false, jres=nothing, S_ij=nothing)
# In python
# ff = SerialDifferentiableFunction(u-> euler_residuals(f,g,s,u.reshape(sh_x),ddr,dprocess,parms,...
#                                with_jres=False,set_dr=False).reshape((-1,sh_x[2])))
# but u.reshape(sh_x) is exactly x, so no need...

res, dres = ff(x)
dres



################################################################################



# N_s = size(s,1)
# n_s = size(s,2)
# n_x = size(x,1)
#
# P = dprocess.values
# Q = dprocess.transitions
#
# n_ms = size(P,1)  # number of markov states
# n_mv = size(P,2)  # number of markov variable
#
# res = zeros(n_ms, N_s, n_x)
#
# if with_jres == true
#   if jres== nothing
#     jres = zeros((n_ms,n_ms,N_s,n_x,n_x))
#   end
#   if jreS_ijs== nothing
#     S_ij = zeros((n_ms,n_ms,N_s,n_s))
#   end
# end
#
# i_ms=1
# m_prep = [repmat(P[i_ms,:]',1) for i in 1:N]
# m=(hcat([e' for e in m_prep]...))'
# xm = x[i_ms]
#
# I_ms=1
# M_prep = [repmat(P[I_ms,:]', 1) for i in 1:N]
# M=(hcat([e' for e in M_prep]...))'
# prob = Q[i_ms, I_ms]
#
# S =
# g(model, m, s, xm, M, parms)
# f(model, m,s,xm,M,s,xm,parms)
# XM = ddr(I_ms, S)
#
# if with_jres==true
#     ff = SerialDifferentiableFunction(u->f(model, m,s,xm,M,S,u,parms))
#     rr, rr_XM = ff(XM)
#     jres[i_ms,I_ms,:,:,:] = prob*rr_XM
#     S_ij[i_ms,I_ms,:,:] = S
# else
#     rr = f(model, m,s,xm,M,S,XM,parms)
#     res[i_ms,:,:] += prob*rr
# end


## Compute derivatives
