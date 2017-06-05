
path = Pkg.dir("Dolo")

# Pkg.build("QuantEcon")
import Dolo
using AxisArrays
include("bruteforce_help.jl")
include("ITI_function.jl")

###############################################################################
filename = joinpath(path,"examples","models","rbc_dtcc_mc.yaml")
# model = Dolo.Model(Pkg.dir("Dolo", "examples", "models", "rbc_dtcc_mc.yaml"), print_code=true)
model = Dolo.yaml_import(filename)

dprocess = Dolo.discretize( model.exogenous )
init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
# improved_time_iteration(model, dprocess, init_dr)

# typeof(model)<:Dolo.AbstractModel
# dr_ITI  = improved_time_iteration(model)
@time dr_ITI  = improved_time_iteration(model)

@time dr_ITI_2  = improved_time_iteration(model, dprocess,init_dr)

@time dr_TI  = Dolo.time_iteration(model)
