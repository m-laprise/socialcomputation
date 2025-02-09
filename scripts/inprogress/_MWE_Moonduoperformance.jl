using Flux
using BenchmarkTools

# Create random inputs and targets
const MINIBATCHSIZE = 64
X = rand(Float32, 100, MINIBATCHSIZE)
Y = rand(Float32, 20, MINIBATCHSIZE)

# Create trivial Flux NN and loss 
model = Chain(Dense(100, 50, relu), 
              Dense(50, 20))
myloss(m, x, y) = Flux.mse(m(x), y)

# Compare time to first gradient:

using Zygote
@btime loss, grads = Flux.withgradient($myloss, $model, $X, $Y)
# 81.875 μs (87 allocations: 126.46 KiB)

using Enzyme
@btime loss, grads = Flux.withgradient($myloss, $Duplicated(model), $X, $Y)
# 82.875 μs (129 allocations: 84.27 KiB)

using Fluxperimental, Mooncake
@btime loss, grads = Flux.withgradient($myloss, $Moonduo(model), $Moonduo(X), $Moonduo(Y))
# 837.625 μs (16045 allocations: 1.89 MiB)

fclosure(m) = myloss(m, X, Y)
@btime loss, grads = Flux.withgradient($fclosure, $Moonduo(model))
# 919.000 μs (16048 allocations: 1.89 MiB)
