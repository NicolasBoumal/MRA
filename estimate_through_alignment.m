function [x_est, X] = estimate_through_alignment(X, x_ref)
% Given observations of a signal as the columns of X, and a reference
% signal x_ref, aligns each observation in X to the signal x_ref using a
% cyclic shift, then averages the aligned observations.
%
% This can be used to simulate an align-to-truth oracle, by supplying the
% correct signal as x_ref (this cannot be done in practice, of course, and
% is used only as a comparison.) Note that such an estimator is biased.
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    [N, M] = size(X);
    assert(length(x_ref) == N);
    x_ref = x_ref(:);
    
    % Align each signal to x_ref
    parfor m = 1 : M
        X(:, m) = align_to_reference(X(:, m), x_ref);
    end
    
    % Average the aligned observations
    x_est = mean(X, 2);

end
