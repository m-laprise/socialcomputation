using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile, DiffResults

# some objective function to work with
f(a, b) = sum(a' * b + a * b')
# pre-record a GradientTape for `f` using inputs of shape 100x100 with Float64 elements
const f_tape = GradientTape(f, (rand(100, 100), rand(100, 100)))
# compile `f_tape` into a more optimized representation
const compiled_f_tape = compile(f_tape)

# some inputs and work buffers to play around with
a, b = rand(100, 100), rand(100, 100)
inputs = (a, b)
results = (similar(a), similar(b))
all_results = map(DiffResults.GradientResult, results)
cfg = GradientConfig(inputs)

####################
# taking gradients #
####################

# with pre-recorded/compiled tapes (generated in the setup above) #
#-----------------------------------------------------------------#

# this should be the fastest method, and non-allocating
gradient!(results, compiled_f_tape, inputs)

# the same as the above, but in addition to calculating the gradients, the value `f(a, b)`
# is loaded into the the provided `DiffResult` instances (see DiffResults.jl documentation).
gradient!(all_results, compiled_f_tape, inputs)

# this should be the second fastest method, and also non-allocating
gradient!(results, f_tape, inputs)

# you can also make your own function if you want to abstract away the tape
∇f!(results, inputs) = gradient!(results, compiled_f_tape, inputs)

# with a pre-allocated GradientConfig #
#-------------------------------------#
# these methods are more flexible than a pre-recorded tape, but can be
# wasteful since the tape will be re-recorded for every call.

gradient!(results, f, inputs, cfg)

gradient(f, inputs, cfg)

# without a pre-allocated GradientConfig #
#----------------------------------------#
# convenient, but pretty wasteful since it has to allocate the GradientConfig itself

gradient!(results, f, inputs)

gradient(f, inputs)

#=================================#

using DifferentiationInterface
import ReverseDiff, Enzyme, Zygote  # AD backends you want to use 

f(x) = sum(abs2, x)

x = [1.0, 2.0]

value_and_gradient(f, AutoForwardDiff(), x) # returns (5.0, [2.0, 4.0]) with ForwardDiff.jl
value_and_gradient(f, AutoEnzyme(),      x) # returns (5.0, [2.0, 4.0]) with Enzyme.jl
value_and_gradient(f, AutoZygote(),      x) # returns (5.0, [2.0, 4.0]) with Zygote.jl

# choose a backend
backend = AutoReverseDiff()
# test
DifferentiationInterface.value_and_gradient(f, backend, x) 

# prepare the gradient calculation
#   preparation does not depend on the actual components of the vector x, just on its type and size
prep = DifferentiationInterface.prepare_gradient(f, backend, zero(x))
# pre allocate
grad = similar(x)
# compute
#   every positional argument between the function and the backend is mutated.
# DifferentiationInterface.gradient!(f, grad, backend, x)
DifferentiationInterface.gradient!(f, grad, prep, backend, x)
grad

y, grad = DifferentiationInterface.value_and_gradient!(f, grad, prep, backend, x)

#==========================#

mutable struct Bar
    baz
    qux::Float64
end

bar = Bar("Hello", 1.5);
bar.qux = 2.0
bar.baz = 1//2

struct Foo
    baz
    qux::Array{Float64}
end

foo = Foo("Hello", [1.5]);
foo.qux .= [2.0]
foo.qux .= 2.0
foo.qux .+= 2.0
foo.qux[1,1] = 2.0
foo.qux[:,:] = [3.0]