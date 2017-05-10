% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

%% Check that Manopt is available
if ~exist('manopt_version', 'file')
    error(sprintf(['Please get Manopt 3.0 or later,\n' ...
                   'available from http://www.manopt.org.\n' ...
                   'Then, run importmanopt.m to add it to the path.'])); %#ok<SPERR>
end

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
M = 25000;
sigma = 2;

tic_generate = tic();
X_data = generate_observations(x_true, M, sigma);
fprintf('Data generation: %.2g [s]\n', toc(tic_generate));

%% Estimate x from the data through invariant features

tic_invfeatmra = tic();

% Estimate the invariants once (this is common to many methods)
tic_invariants = tic();
[mean_est, P_est, B_est] = invariants_from_data(X_data, sigma);
fprintf('Estimation of invariant features: %.2g [s]\n', toc(tic_invariants));

% Estimate the DFT phases from the bispectrum (this can be done with
% different algorithms, and the non-convex approach
% phases_from_bispectrum_real can accept an input z_init to initialize.)
% We could call many different algorithms here to compare.
tic_bispectrum_inversion = tic();
z_est = phases_from_bispectrum_real(B_est);
fprintf('Estimation of phases from bispectrum: %.2g [s]\n', toc(tic_bispectrum_inversion));

% Recombine (this is computationally cheap)
x_est = combine_features(mean_est, P_est, z_est);

time_invfeatmra = toc(tic_invfeatmra);

% This is a shortcut that does all in one (easier for end user but not
% convenient for doing many tests and comparisons):
% tic_mra = tic();
% x_est = MRA_invariant_features(X, sigma);
% fprintf('Estimation from data: %.2g [s]\n', toc(tic_mra));

%% It is informative to compare against EM
fprintf('Running EM (this could take a few minutes.)\n');
tic_em = tic();
x_em = MRA_EM(X_data, sigma);
time_em = toc(tic_em);
fprintf('Estimation of signal via expectation maximization: %.2g [s]\n', time_em);

%% Evaluate error, and plot
relerr = relative_error(x_true, x_est); % 2-norm, up to integer shifts
relerr_em = relative_error(x_true, x_em); % 2-norm, up to integer shifts
% Align for plotting
x_est = align_to_reference(x_est, x_true);
x_em = align_to_reference(x_em, x_true);
hold all;
t = 0:(N-1);
plot(t, x_true, '.-');
plot(t, x_est, 'o-');
plot(t, x_em, 'o-');
title(sprintf('Multireference alignment example, M = %d, sigma = %g', M, sigma));
legend('True signal', ...
       sprintf('Invariant features (RMSE: %.2g; time: %.2g [s])', relerr, time_invfeatmra), ...
       sprintf('Expectation maximization (RMSE: %.2g; time: %.2g [s])', relerr_em, time_em));
xlim([0, N-1]);
