function x_est = combine_features(mean_est, P_est, z_est)
% Combine estimates for the mean, the power spectrum and the phases of the
% DFT of a signal x in order to estimate it. The estimated signal is
% returned.
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    N = length(P_est);
    assert(length(z_est) == N, 'P_est and z_est must have same length.');

    x_fft_est = sqrt(P_est) .* z_est;
    x_fft_est(1) = mean_est*N;
    x_est = ifft(x_fft_est);

end
