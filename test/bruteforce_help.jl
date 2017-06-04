


function euler_residuals(f, g, s::AbstractArray, x::Array{Array{Float64,2},1}, dr,
                         dprocess, parms::AbstractArray; with_jres=false, set_dr=true,
                         jres=nothing, S_ij=nothing)

    if set_dr ==trues
      Dolo.set_values!(dr,x)
    end

    N_s = size(s,1) # Number of gris points for endo_var
    n_s = size(s,2) # Number of states
    n_x = size(x,1) # Number of controls

    P = dprocess.values
    Q = dprocess.transitions

    n_ms = size(P,1)  # number of markov states
    n_mv = size(P,2)  # number of markov variable

    res = zeros(n_ms, N_s, n_x)

    if with_jres == true
      if jres== nothing
        jres = zeros((n_ms,n_ms,N_s,n_x,n_x))
      end
      if S_ij== nothing
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
          end
          res[i_ms,:,:] += prob*rr
        end

    end

    res_AA = AxisArray(res, Axis{:n_m}(1:n_ms), Axis{:N_s}(1:N_s), Axis{:n_x}(1:n_x))
    if with_jres==true
        return res_AA, jres, S_ij
    else
        return res_AA
    end

end
#############################################################################
# I am still not sure we need it
function euler_residuals(f, g, s::AbstractArray, x::Array{Float64,2}, dr,
                         dprocess, parms::AbstractArray; with_jres=false, set_dr=true,
                         jres=nothing, S_ij=nothing)
   N_m = Dolo.n_nodes(dprocess.grid)
   x_reshaped = Dolo.destack0(xi,N_m)
   return euler_residuals(f,g,s,x_reshaped,dr,dprocess,parms; kwargs...)
end



######################
function SerialDifferentiableFunction(f, epsilon=1e-8)

    function df(x::AbstractVector)

      v0 = f(x)

      n_m = size(v0,1)
      N_s = size(v0,2)
      n_v = size(v0,3)
      assert(size(x[1],1) == N_s)
      n_x = size(x,1)

      dv = zeros(n_m*N_s,  n_v, n_x)
      for i in 1:n_x
      xi = deepcopy(cat(1,x...))
      xi[:,i] += epsilon
      # You could also use f(xi)
      vi = f(Dolo.destack0(xi,n_m))
      dd=(vi+(-1*v0))./epsilon
      # dd=permutedims(dd, [2,1,3])
      dv[:,:, i] = reshape(dd,N_s*n_m,n_x) # (1dim) corresponds to equations, in raws you first stuck derivatives wrt 1rst exo state, 2nd, etc
      end
      dv_AA = AxisArray(dv, Axis{:N}(1:n_m*N_s), Axis{:n_v}(1:n_v), Axis{:n_x}(1:n_x))
        return [v0, dv_AA]
    end


    function df(x::Array{Float64,2})

      v0 = f(x)

      # n_m = size(v0,1)
      N_s = size(v0,1)
      n_v = size(v0,2)
      assert(size(x,1) == N_s)
      n_x = size(x,2)

      dv = zeros(N_s,  n_v, n_x)
      for i in 1:n_x
         xi = deepcopy(x)
         xi[:,i] += epsilon
         vi = f(xi)
         dd=(vi+(-1*v0))./epsilon
         dv[:,:, i] = dd
      end
      dv_AA = AxisArray(dv, Axis{:N}(1:N_s), Axis{:n_v}(1:n_v), Axis{:n_x}(1:n_x))
        return [v0, dv_AA]
    end

    df
end




function swaplines(i,j,M::Array{Float64,3})
    n0, n1, n2 = size(M)
    for k in 1:n1
        for l in 1:n2
            t = M[i,k,l]
            M[i,k,l] = M[j,k,l]
            M[j,k,l] = t
        end
    end
    return M
end

function swaplines(i,j,M::Array{Float64,2})
    n = size(M,2)
    for k in 1:n
        t = M[i,k]
        M[i,k] = M[j,k]
        M[j,k] = t
    end
    return M
end

function swaplines(i,j,M::Array{Float64,1})
    n = size(M,1)
    t = M[i]
    M[i] = M[j]
    M[j] = t
    return M
end



function divide(i,c,M::Array{Float64,3})
    n0, n1, n2 = size(M)
    for k in 1:n1
        for l in 1:n2
            M[i,k,l] /= c
        end
    end
    return M
end

function divide(i,c,M::Array{Float64,2})
    n = size(M,1)
    for k in 1:n
        M[i,k] /= c
    end
    return M
end



function divide(i,c,M::Array{Float64,1})
    M[i] /= c
    return M
end


function substract(i,j,c,M::Array{Float64,3})
    n0, n1, n2 = size(M)
    for k in 1:n1
        for l in r1:n2
            M[i,k,l] = M[i,k,l] - c*M[j,k,l]
        end
    end
    return M
end

function substract(i,j,c,M::Array{Float64,2})
    # Li <- Li - c*Lj
    n = size(M,1)
    for k in 1:n
        M[i,k] = M[i,k] - c*M[j,k]
    end
    return M
end
function substract(i,j,c,M::Array{Float64,1})
    M[i] = M[i] - c*M[j]
    return M
end



function invert(A,B)
   for i in 1:size(A,1)
         max_err = -1.0
         max_i = 0

         for i0 in 1:size(A,1)
            err = abs(A[i0,i])
            if err>=max_err
                max_err = err
                max_i = i0
            end
         end

         swaplines(i,max_i,A)
         swaplines(i,max_i,B)
         c = A[i,i]
         divide(i,c,A)
         divide(i,c,B)

         for i0 in i+1:size(A,1)
             f = A[i0,i]
             substract(i0,i,f,A)
             substract(i0,i,f,B)
         end
   end

   for i in size(A,1):-1:0
       for i0 in 1:i-1
           f = A[i0,i]
           substract(i0,i,f,A)
           substract(i0,i,f,B)
       end
   end
   return A, B
end


function ssmul(A,B)
    # simple serial_mult (matrix times vector)
    N,a,b = size(A)
    NN,b = size(B)
    O = zeros(N,a)
    for n in 1:N
        for k in 1:a
            for l in 1:b
                O[n,k] += A[n,k,l]*B[n,l]
            end
        end
    end
    return O

end

function destack0(x::Array{Float64,3},n_m::Int)
   xx=collect(x)
   return [xx[i, :, :] for i=1:n_m]
end

function d_filt_dx(res,jres,S_ij,n_m,N,n_x,dumdr; precomputed=false)

    # xx=collect(res)
    # res_m = [xx[i, :, :] for i=1:n_m]

    Dolo.set_values!(dumdr,destack0(res, n_m))
    i = 1
    j = 1
    for i in 1:n_m
        res[i,:,:] = 0
        for j in 1:n_m
            A = jres[i,j,:,:,:]
            if precomputed== false
                B = dumdr(j,S_ij[i,j,:,:])
            else
                B = dumdr(i,j) #,fut_S[i,j,:,:])
            end
            res[i,:,:] += ssmul(A,B)
        end
    end
    res
    return res
end


function invert_jac(res,dres,jres,fut_S; filt= nothing, tol=1e-10, maxit=1000, verbose=false)
    n_m, N, n_x = size(res)
    err0 = 0.0
    ddx = zeros(n_m,N_s,n_x)
    A=deepcopy(dres)
    B = deepcopy(res)
    for i_m in 1:n_m
        for n in 1:N_s
           ddx[i_m,n,:]= invert(A[i_m,n,:,:],collect(B[i_m,n,:]))[2]
        end
    end
    filt = ddr_filt
    if filt == nothing
      error("No filter supplied.")
    else
      dumdr = ddr_filt
    end
    lam = -1.0
    lam_max = -1.0
    err_0 = abs(maximum(ddx))
    tot = deepcopy(ddx)
    verbose=true
    if verbose==true
    print("Starting inversion")
    end
    maxit=1000
    err=err_0
    tol=1e-08
    verbose && @printf "%-6s%-12s%-5s\n" "err" "lam" "lam_max"
    it = 0
    while it<maxit && err>tol
      it +=1
      precomputed=false
      dumdr

      ddx = d_filt_dx(ddx,jres,fut_S,n_m,N,n_x,dumdr; precomputed=precomputed)
      # might also work
      # d_filt_dx(ddx,jres,fut_S,n_m,N,n_x,dumdr; precomputed=precomputed)

      err = abs(maximum(ddx))
      lam = err/err_0
      lam_max = max(lam_max, lam)
      verbose && @printf "%-6s%-12s%-5s\n" err lam lam_max
      # if verbose==true
      #     print('- {} | {} | {}'.format(err, lam, lam_max))
      # end
      tot += ddx
      err_0 = err
    end
    tot += ddx*lam/(1-lam)
    return tot, it, lam
end


type ImprovedTimeIterationResult
    # dr::AbstractDecisionRule
    N::Int
    f_x::Float64
    d_x::Float64
    # Time_residuals::
    # Time_inversion::
    # Time_search::
    tol::Float64
    Lambda::Float64
    N_invert::Float64
    N_search::Float64
end

converged(r::ImprovedTimeIterationResult) = r.f_x

function Base.show(io::IO, r::ImprovedTimeIterationResult)
    @printf io "Results of Improved Time Iteration Algorithm\n"
    @printf io " * Number of iterations: %s\n" string(r.N)
    # @printf io " * Complementarities: %s\n" string(r.complementarities)
    # @printf io " * Decision Rule type: %s\n" string(typeof(r.dr))
    @printf io " * Convergence: %s\n" converged(r)
    @printf io " * Contractivity: %s\n" string(r.Lambda)
    @printf io "   * |x - x'| < %.1e: %s\n" r.tol r.f_x
end
