% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

%% Generate a signal x of length N
N = 101;
x_true = zeros(N, 1);
x_true(31:60) = 1;

%% Start the parallel pool
parallel_nodes = 2;
if isempty(gcp('nocreate'))
    parpool(parallel_nodes, 'IdleTimeout', 240);
end

%% Generate M observations with noise variance sigma^2
M = 10000;
sigma = 1;

tic_generate = tic();
X_data = generate_observations(x_true, M, sigma);
fprintf('Data generation: %.2g [s]\n', toc(tic_generate));

%% Estimate x from the data through invariant features

% Estimate the invariants once (this is common to many methods)
tic_invariants = tic();
[mean_est, P_est, B_est] = invariants_from_data(X_data, sigma);
fprintf('Estimation of invariant features: %.2g [s]\n', toc(tic_invariants));

% Estimate the DFT phases from the bispectrum (this can be done with
% different algorithms, and the non-convex approach
% phases_from_bispectrum_real can accept an input z0 to initialize.)
% We could call many different algorithms here to compare.
tic_bispectrum_inversion = tic();
z_est = phases_from_bispectrum_real(B_est);
fprintf('Estimation of phases from bispectrum: %.2g [s]\n', toc(tic_bispectrum_inversion));

% Recombine (this is computationally cheap)
x_est = combine_features(mean_est, P_est, z_est);

% This is a shortcut that does all in one (easier for end user but not
% convenient for doing many tests and comparisons):
% tic_mra = tic();
% x_est = MRA_invariant_features(X, sigma);
% fprintf('Estimation from data: %.2g [s]\n', toc(tic_mra));

%% It is informative to compare against EM
tic_em = tic();
x_em = MRA_EM(X_data, sigma);
fprintf('Estimation of signal via expectation maximization: %.2g [s]\n', toc(tic_em));

%% Evaluate error, and plot
relerr = relative_error(x_true, x_est); % up to integer shifts
relerr_ora = relative_error(x_true, x_em); % up to integer shifts
t = 0:(N-1);
% Align for plotting
x_est = align_to_reference(x_est, x_true);
x_em = align_to_reference(x_em, x_true);
hold all;
plot(t, x_true, '.-');
plot(t, x_est, 'o-');
plot(t, x_em, 'o-');
title(sprintf('Relative error: %.2e (EM: %.2e)', relerr, relerr_ora));
legend('signal', 'invariant features MRA', 'expectation maximization');
xlim([0, N-1]);
