import Dolo

model = Dolo.Model(Pkg.dir("Dolo","examples","models","rbc_dtcc_mc.yaml"))

Dolo.time_iteration(model)
dr0 = Dolo.time_iteration(model, details=false, tol_η=1e-6; grid=Dict(:n=>[5]))
sol = Dolo.time_iteration(model, maxit=1000; grid=Dict(:n=>[100]))

# faster version: limit the number of steps in the newton solver
@time sol = Dolo.time_iteration(model, maxit=1000; solver=Dict(:maxit=>2), grid=Dict(:n=>[100]))

# or use explicit formulas given in the model
@time sol = Dolo.time_iteration(model, maxit=1000; solver=Dict(:type=>:direct), grid=Dict(:n=>[100]))
