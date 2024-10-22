using GLM
using DataFrames
using Random
using Distributions

# For each pair of X, y: sample a training set and a test set. 
# Fit a linear model on the training set and evaluate on the test set. 
# Report model R-squared and average test error.
# Compare with the null model that predicts the mean of y.
function fit_linear_model(X, y)
    n = size(X, 1)
    ntrain = Int(round(n * 0.8))
    ntest = n - ntrain
    permuted = randperm(n)
    trainidx = permuted[1:ntrain]
    testidx = permuted[ntrain+1:end]
    Xtrain = X[trainidx, :]
    ytrain = y[trainidx]
    Xtest = X[testidx, :]
    ytest = y[testidx]
    k = size(X, 2)
    # Create data 
    train_data = DataFrame(y=ytrain)
    for i in 1:k
        train_data[!, Symbol("X$i")] = Xtrain[:, i]
    end
    test_data = DataFrame(y=ytest)
    for i in 1:k
        test_data[!, Symbol("X$i")] = Xtest[:, i]
    end
    formula = Term(:y) ~ sum(Term(Symbol("X$i")) for i in 1:k)
    model = lm(formula, train_data)
    rsquared = r2(model)
    ypred = predict(model, test_data)
    testerror = sum((ytest - ypred).^2) / ntest
    return rsquared, testerror, model
end