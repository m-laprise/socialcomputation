


function bias_process(alpha::Float64, start::Int, stop::Int, maxsteps::Int)
    biasproc = zeros(maxsteps)
    biasproc[start:stop] .= alpha
    return biasproc
end
