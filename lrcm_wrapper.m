% Wrapper file for the LRCM MIN template of Pietersz and Grubisic.
% LRCM MIN with the weighted Hadamard objective function find the nearest
% low-rank correlation matrix to a given input.
% See `weighted distance min` folder for example function and derivatives.
function Fopt = lrcm_wrapper()
%    [minimum,optimal_matrix] = lrcm_min(initial_guess)
load('ml_tmp.mat');
global FParameters;
FParameters.C = C;
FParameters.W = W;
[Fopt,optimal_matrix] = lrcm_min(Y0);
save('py_tmp.mat','optimal_matrix');
