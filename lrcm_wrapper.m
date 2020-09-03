% Wrapper file for the LRCM MIN template of Pietersz and Grubisic.
% LRCM MIN with the weighted Hadamard objective function find the nearest
% low-rank correlation matrix to a given input.
% See `weighted distance min` folder for example function and derivatives.
function optimal_matrix = lrcm_wrapper(weights,cost,d)
    global FParameters;
    FParameters.C = cost
    FParameters.W = weights
    [minimum,optimal_matrix] = lrcm_min(guess(cost,d))
