function x_est = MRA_invariant_features(X, sigma)
% Full algorithm to estimate a signal x of length N from M noisy,
% circularly shifted observations of x. The observations are stored in the
% matrix X, of size N x M. The noise standard deviation is given as sigma.
%
% The algorithm goes through two steps:
%  1) It estimates features of x which are invariant under circular shifts,
%     namely, the mean, the power spectrum and the bispectrum.
%  2) It estimates the phases of the DFT of the signal from the bispectrum.
% Then, the estimated mean, power spectrum and DFT phases can be combined
% into an estimator of x.
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    [mean_est, P_est, B_est] = invariants_from_data(X, sigma);

    if isreal(X)
        z_est = phases_from_bispectrum_real(B_est, sign(mean_est));
    else
        z_est = phases_from_bispectrum_complex(B_est);
    end
    
    x_est = combine_features(mean_est, P_est, z_est);

end
