function [z, problem] = phases_from_bispectrum_real(B_weighted, sign_mean, z_init)
% Given a (possibly weighted) bispectrum of a real signal, estimates the
% phases of the DFT of that signal via non-convex optimization with Manopt.
% z_init (optional) is an initial guess for the phases (random if omitted.)
% sign_mean is the sign of the mean of the signal (+1 or -1). It will be
% estimated from the bispectrum if omitted.
%
% Requires Manopt: http://www.manopt.org/
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    % The mean of a real signal has sign +1 or -1 (if the mean is 0, we
    % assimilate that to sign +1). As a consequence, the manifold of phases
    % of the DFTs of real signals is disconnected, and we must pick the
    % appropriate component.
    if ~exist('sign_mean', 'var') || isempty(sign_mean)
        sign_mean = sign(B_weighted(1, 1));
    end
    if sign_mean == 0
        sign_mean = 1;
    end
    assert(sign_mean == 1 || sign_mean == -1, 'sign_mean must be +1 or -1.');

    N = size(B_weighted, 1);
    assert(size(B_weighted, 2) == N, 'B must be square.');

    Mfun = @(z) B_weighted .* conj(circulant(z));
    
    problem.M = realphasefactory(N, sign_mean, 1);
    problem.cost  = @(z) -real(z'*Mfun(z)*z) / N^2;
    problem.egrad = @(z) -3*Mfun(z)*z / N^2;
    problem.ehess = @(z, zdot) -6*Mfun(z)*zdot / N^2;

    % Allow initialization of the non-convex optimization.
    % If no z_init is specified, a random guess is generated.
    if ~exist('z_init', 'var')
        z_init = [];
    else
        % Ensure the initial guess contains phases only,
        % that these phases have the proper symmetry for real signals,
        % and, if N is even, make sure the middle frequency phase has same
        % sign in z_init and in the manifold definition.
        assert(length(z_init) == N, 'z_init must have length N.');
        assert(all(z_init ~= 0), 'Initial guess must contain phases.');
        z_init = sign(problem.M.downup(z_init));
        if N/2 == round(N/2)
            problem.M = realphasefactory(N, sign_mean, sign(z_init(N/2+1)));
        end
    end
    
    % Actual optimization happens here.
    options.tolgradnorm = 1e-8;
    options.verbosity = 0;
    [z, zcost] = trustregions(problem, z_init, options);

    % When N is even, even in the noiseless case we know the cost function
    % can have strict local optima. An easy way to escape them (in the
    % noiseless case) is to flip the sign of the middle frequency, and to
    % reoptimize from there.
    if N/2 == round(N/2)
        z2 = z;
        z2(N/2 + 1) = -z2(N/2 + 1);
        z2cost = problem.cost(z2);
        if z2cost < zcost
            if options.verbosity > 0
                fprintf('Flipping z[N/2]\n');
            end
            problem.M = realphasefactory(N, sign_mean, z2(N/2 + 1));
            z = trustregions(problem, z2, options);
        end
    end

end
