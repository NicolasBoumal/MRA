function [X, shifts] = generate_observations(x, M, sigma, noisetype)
% Given a signal x of length N, generates a matrix X of size N x M such
% that each column of X is a randomly, circularly shifted version of x with
% i.i.d. Gaussian noise of variance sigma^2 added on top. If x is complex,
% the noise is also complex.
%
% If noisetype is set to 'uniform' instead of 'Gaussian' (or instead of
% being omitted), the noise is uniformly distributed in a centered interval
% such that the variance is sigma^2. (For real signals only.)
%
% The optional second output, shifts, contains the true shifts that affect
% the individual columns of X: M integers between 0 and N-1.
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    x = x(:);
    N = length(x);
    
    X = zeros(N, M);
    shifts = randi(N, M, 1) - 1;
    for m = 1 : M
        X(:, m) = circshift(x, shifts(m));
    end
    
    if ~exist('noisetype', 'var') || isempty(noisetype)
        noisetype = 'Gaussian';
    end
    
    switch lower(noisetype)
        case 'gaussian'
            if isreal(x)
                X = X + sigma*randn(N, M);
            else
                X = X + sigma*(randn(N, M) + 1i*randn(N, M))/sqrt(2);
            end
        case 'uniform'
            if isreal(x)
                X = X + sigma*(rand(N, M)-.5)*sqrt(12);
            else
                error('Uniform complex noise not supported yet.');
            end
        otherwise
            error('Noise type can be ''Gaussian'' or ''uniform''.');
    end

end
