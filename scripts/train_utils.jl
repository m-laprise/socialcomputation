#=
The script includes functions required to run the training loop under various experimental conditions.
It also includes helper functions for inspecting and diagnosing gradients during training.
=#

#### TRAINING UTILITIES

"""
    getjacobian(activemodel; x=nothing, wrt="state")

Compute the Jacobian of a model with respect to its input or state.

# Arguments
- `activemodel`: The model to compute the Jacobian for.
- `x`: The input to the model. Required if `wrt="input"`.
- `wrt`: The type of Jacobian to compute. Can be either "input" or "state".

# Returns
- The Jacobian matrix. Its size is `(output_size, input_size)` if `wrt="input"`, or
  `(state_size, state_size)` if `wrt="state"`.
"""
function getjacobian(activemodel; x=nothing, wrt="state")
    if wrt == "input"
        @assert !isnothing(x)
        J = Zygote.jacobian(activemodel, x)[1]
    elseif wrt == "state"
        h = state(activemodel.layers[1])
        x = zeros(Float32, length(h))
        J = Zygote.jacobian(x) do y
            activemodel(y); state(activemodel.layers[1])
        end
    else
        error("Invalid wrt argument. Choose from 'input' or 'state'.")
    end
    return J[1]
end

"""
    inspect_gradients(grads)

Inspect the gradients for NaN, vanishing, or exploding values during the training loop
and return the number of parameters with these issues.
"""
function inspect_gradients(grads)
    g, _ = Flux.destructure(grads)
    
    nan_params = [0]
    vanishing_params = [0]
    exploding_params = [0]

    if any(isnan.(g))
        push!(nan_params, 1)
    end
    if any(abs.(g) .< 1e-6)
        push!(vanishing_params, 1)
    end
    if any(abs.(g) .> 1e6)
        push!(exploding_params, 1)
    end
    return sum(nan_params), sum(vanishing_params), sum(exploding_params)
end

"""
    diagnose_gradients(n, v, e)

Takes as input the output of `inspect_gradients` and prints a message based on the number of
parameters with NaN, vanishing, or exploding gradients.
"""
function diagnose_gradients(n, v, e)
    if n > 0
        println(n, " NaN gradient detected")
    end
    if v > 0
        println(v, " vanishing gradient detected")
    end
    if e > 0
        println(e, " exploding gradient detected")
    end
    #Otherwise, report that no issues were found
    if n == 0 && v == 0 && e == 0
        println("Gradients appear to be well-behaved.")
    end
end

### TRAINING functions

