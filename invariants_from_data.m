function [mean_est, P_est, B_est] = invariants_from_data(X, sigma)
% Given M noisy, circularly shifted observations of a signal of length N in
% a matrix X of size N x M and the standard deviation sigma of the noise,
% returns an estimate of the mean, the power spectrum and the bispectrum of
% the signal. Uses a parfor for the bispectrum.
%
% For the power spectrum, the noise power is subtracted to avoid bias. If
% this results in negative entries, those are trimmed to zero.
%
% For the bispectrum, it is an estimate of the bispectrum of the centered
% signal (mean removed) to avoid bias. This operation affects only the
% first row and column of B as well as its diagonal.
%
% If sigma is not specified, it will be estimated from the data.
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    [N, M] = size(X);
    
    %% If sigma is not supplied, estimate it from the data
    if ~exist('sigma', 'var') || isempty(sigma)
        sigma = std(sum(X, 1))/sqrt(N);
    end

    %% Estimate the mean and subtract it from the observations.
    %  This also gives an estimate of the first component of the fft of x.
    %  Centering the observations helps for the bispectrum estimation part.
    mean_est = mean(X(:));
    Xc = X - mean_est;

    %% Prepare DFTs of centered signals
    Xc_fft = fft(Xc);

    %% Estimate the power spectrum (gives estimate of modulus of DFT of x).
    P_est = mean(abs(Xc_fft).^2, 2);
    % Debias the estimate using the supplied or estimated value of sigma
    P_est = P_est - N*sigma^2;
    % Take the nonnegative part
    P_est = max(0, P_est);
    
    % Note: it does not matter whether the debiasing is applied to the DC
    % component or not, since eventually the DC component of x is estimated
    % from the estimation of the mean.

    %% Estimate the bispectrum
    if nargout >= 3
        
        B_est = zeros(N);
        parfor m = 1 : M
            xm_fft = Xc_fft(:, m);
            Bm = (xm_fft*xm_fft') .* circulant(xm_fft);
            B_est = B_est + Bm;
        end
        B_est = B_est/M;
        
    end

end

% Note:
% 
% If M is extremely large, then computing these averages results in
% a large numerical error due to finite arithmetic precision. If
% this error surpasses the statistical estimation error, we will
% see saturation/decline in the estimation quality as M continues
% to grow. This never appeared in past experiments, but is good to keep in
% mind if one pushes M to extremes.



% I tried looping over the entries of the bispectrum rather than
% over the observations, but this turns out to be much slower when
% parpool has a lot of workers. Turning either of the for loops
% below into a parfor is even slower because the matrix Xc_fft_t
% has to be broadcast entirely to all workers (or at least, Matlab
% sees no other way of doing it with the code provided.)
% Still might be useful if we want to compute only a sparse subset
% of the bispectrum.
%
% % B_est2 = zeros(N);
% % Xc_fft_t = Xc_fft.';
% % for k1 = 1 : N
% %     for k2 = 1 : N
% %         k2mk1 = 1 + mod(k2 - k1, N);
% %         a = Xc_fft_t(:, k1);
% %         b = Xc_fft_t(:, k2);
% %         c = Xc_fft_t(:, k2mk1);
% %         B_est2(k1, k2) = mean(a.*conj(b).*c);
% %     end
% % end
