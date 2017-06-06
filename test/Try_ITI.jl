
path = Pkg.dir("Dolo")

# Pkg.build("QuantEcon")
import Dolo
import Bruteforce_module
# using AxisArrays
# include("bruteforce_help.jl")
# include("ITI_function.jl")

###############################################################################
filename = joinpath(path,"examples","models","rbc_dtcc_mc.yaml")
# model = Dolo.Model(Pkg.dir("Dolo", "examples", "models", "rbc_dtcc_mc.yaml"), print_code=true)
model = Dolo.yaml_import(filename)

@time dr_ITI  = Bruteforce_module.improved_time_iteration(model; compute_radius=true)

# dprocess = Dolo.discretize( model.exogenous )
# init_dr = Dolo.ConstantDecisionRule(model.calibration[:controls])
# @time dr_ITI_2  = Bruteforce_module.improved_time_iteration(model, dprocess,init_dr)
#
# @time dr_TI  = Dolo.time_iteration(model)


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












# unstable(x) = x > 0.5 ? true : 0.0
#
# function profile_unstable_test(m, n)
#     s = s2 = 0
#     for i = 1:n
#         for k = 1:m
#             s += unstable(rand())
#         end
#         x = collect(1:20)
#         s2 += sum(x)
#     end
#     s, s2
# end
#
# profile_unstable_test(1, 1)
# Profile.clear()
# @profile profile_unstable_test(10, 10^6)
# ProfileView.view()
#
