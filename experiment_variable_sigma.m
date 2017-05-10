% This file can be run from line 120 onward if data is available.

%% Clean up
clear;
close all;
clc;

%% Create a real signal
N = 41;
x_true = zeros(N, 1);
x_true(5:25) = 1;

%% Experiment
M = 1e4;
sigmas = logspace(-2, 1, 13);
repeats = 20;
metric = zeros(7, length(sigmas), repeats);
cptime = zeros(7, length(sigmas), repeats);

for repeat = 1 : repeats
    for iter = 1 : length(sigmas)

        % Make sure the parallel pool is started
        % (LLL is so slow that parpool sometimes disconnects.)
        if isempty(gcp('nocreate'))
            parpool(30, 'IdleTimeout', 240);
        end
        
        fprintf('Repetition %4d/%d, iteration %2d/%d\n', repeat, repeats, iter, length(sigmas));
        
        % Generate observations
        sigma = sigmas(iter);
        fprintf('\tGenerate observations (sigma = %g)\n', sigma);
        noisetype = 'Gaussian';
        [X, shifts] = generate_observations(x_true, M, sigma, noisetype);

        % Use the invariant feature approach to recover the signal
        fprintf('\tCompute invariants\n');
        tic_invariants = tic();
        [mu, P, B] = invariants_from_data(X, sigma);
        time_invariants = toc(tic_invariants);
        
        cptime(7, iter, repeat) = time_invariants;
        
        % Non-convex approach
        fprintf('\tRTR\n');
        tic_est = tic();
        z_est = phases_from_bispectrum_real(B);
        x_est = combine_features(mu, P, z_est);
        cptime(5, iter, repeat) = time_invariants + toc(tic_est);
        
        % Iterative phase synchronization approach
        fprintf('\tAPS\n');
        tic_aps = tic();
        z_aps = phases_from_bispectrum_APS_real(B, sign(mu));
        x_aps = combine_features(mu, P, z_aps);
        cptime(4, iter, repeat) = time_invariants + toc(tic_aps);
        
        % SDP, with true two first phases (slightly unfair advantage)
        fprintf('\tSDP\n');
        y = fft(x_true);
        tic_sdp = tic();
        z_sdp = phases_from_bispectrum_SDP_real(B, sign(y(1)), sign(y(2)));
        x_sdp = combine_features(mu, P, z_sdp);
        cptime(3, iter, repeat) = time_invariants + toc(tic_sdp);
        
        % FM, with true two first phases (slightly unfair advantage)
        fprintf('\tFM\n');
        y = fft(x_true);
        tic_fm = tic();
        z_fm = phases_from_bispectrum_FM_real(B, sign(y(1)), sign(y(2)));
        x_fm = combine_features(mu, P, z_fm);
        cptime(1, iter, repeat) = time_invariants + toc(tic_fm);
        
        % EM
        fprintf('\tEM\n');
        tic_em = tic();
        x_em = MRA_EM(X, sigma);
        cptime(6, iter, repeat) = toc(tic_em);
        
        % Oracle who knows the shifts
        fprintf('\tOracle\n');
        X_unshifted = zeros(N, M);
        for m = 1 : M
            X_unshifted(:, m) = circshift(X(:, m), -shifts(m));
        end
        x_ora = mean(X_unshifted, 2);
        
        
        % LLL, with true two first phases (slightly unfair advantage)
        if sigma <= 1
            fprintf('\tLLL\n');
            y = fft(x_true);
            tic_lll = tic();
            z_lll = phases_from_bispectrum_LLL_real(B, sign(y(1)), sign(y(2)));
            x_lll = combine_features(mu, P, z_lll);
            cptime(2, iter, repeat) = time_invariants + toc(tic_lll);
            metric(2, iter, repeat) = relative_error(x_true, x_lll);
        end
        

        metric(1, iter, repeat) = relative_error(x_true, x_fm);
        %
        metric(3, iter, repeat) = relative_error(x_true, x_sdp);
        metric(4, iter, repeat) = relative_error(x_true, x_aps);
        metric(5, iter, repeat) = relative_error(x_true, x_est);
        metric(6, iter, repeat) = relative_error(x_true, x_em);
        metric(7, iter, repeat) = relative_error(x_true, x_ora);

    end
end

clear X;
clear X_unshifted;
clear shifts;
save experiment_variable_sigma.mat;



%% Display (you can start here if the data file is available)

load experiment_variable_sigma.mat;

%%

figure(3);
clf;

hold all;
loglog(sigmas, mean(metric(1, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(metric(2, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(metric(3, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(metric(4, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(metric(5, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(metric(6, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(metric(7, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
xlabel('Noise level \sigma');
ylabel('Relative error (up to circular shift)');
hlegend = legend('Invariant features, FM', ...
                 'Invariant features, phase unwrapping', ...
                 'Invariant features, SDP', ...
                 'Invariant features, iter. phase synch.', ...
                 'Invariant features, optim. phase manifold', ...
                 'Expectation maximization', ...
                 'Known-shifts oracle', ...
                 'Location', 'NorthWest');
set(hlegend, 'Box', 'off');
fontsz = 14;
set(hlegend, 'FontSize', fontsz);
% ylim([0, 1]);
xlim([min(sigmas), max(sigmas)]);
% title(sprintf('Fixed signal, N = %d, \\sigma = %g,\naveraged over %d repeats, %s noise', N, sigma, repeats, noisetype));
pbaspect([1.6, 1, 1])

set(gcf, 'Color', 'w');

%%

figure(4);
clf;

hold all;
loglog(sigmas, mean(cptime(1, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(cptime(2, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(cptime(3, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(cptime(4, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(cptime(5, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
loglog(sigmas, mean(cptime(6, :, :), 3), '.-', 'LineWidth', 1.5, 'MarkerSize', 10);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
xlabel('Noise level \sigma');
ylabel('Relative error (up to circular shift)');
hlegend = legend('Invariant features, FM', ...
                 'Invariant features, phase unwrapping', ...
                 'Invariant features, SDP', ...
                 'Invariant features, iter. phase synch.', ...
                 'Invariant features, optim. phase manifold', ...
                 'Expectation maximization', ...
                 'Location', 'NorthWest');
set(hlegend, 'Box', 'off');
fontsz = 14;
set(hlegend, 'FontSize', fontsz);
% ylim([0, 1]);
xlim([min(sigmas), max(sigmas)]);
pbaspect([1.6, 1, 1])

set(gcf, 'Color', 'w');

%%

figure(3);
figname = 'experiment_variable_sigma_RMSE';
pdf_print_code(gcf, [figname, '.pdf'], fontsz);
saveas(gcf, [figname, '.fig']);

figure(4);
figname = 'experiment_variable_sigma_CPU';
pdf_print_code(gcf, [figname, '.pdf'], fontsz);
saveas(gcf, [figname, '.fig']);
