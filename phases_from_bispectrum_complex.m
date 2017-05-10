function [z, problem] = phases_from_bispectrum_complex(B_weighted, z_init)
% Given a (possibly weighted) bispectrum of a complex signal, estimates the
% phases of the DFT of that signal via non-convex optimization with Manopt.
% z_init (optional) is an initial guess for the phases (random if omitted.)
%
% Requires Manopt: http://www.manopt.org/
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    N = size(B_weighted, 1);
    assert(size(B_weighted, 2) == N, 'B must be square.');

    Mfun = @(z) B_weighted .* conj(circulant(z));
    
    problem.M = complexcirclefactory(N);
    problem.cost  = @(z) -real(z'*Mfun(z)*z) / N^2;
    problem.egrad = @(z) -(2*Mfun(z)*z + Mfun(z)'*z)/ N^2;
    problem.ehess = @(z, zdot) -(2*Mfun(zdot)*z + 2*Mfun(z)*zdot + Mfun(zdot)'*z + Mfun(z)'*zdot)/ N^2;
    
    % Allow initialization of the non-convex optimization.
    % If no z_init is specified, a random guess is generated.
    if ~exist('z_init', 'var')
        z_init = [];
    else
        % Ensure the initial guess contains phases only
        assert(length(z_init) == N, 'z_init must have length N.');
        assert(all(z_init ~= 0), 'Initial guess must contain phases.');
        z_init = sign(z_init);
    end
    
    % Actual optimization happens here.
    options.tolgradnorm = 1e-8;
    options.verbosity = 0;
    z = trustregions(problem, z_init, options);

end
