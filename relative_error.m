function relerr = relative_error(x_ref, x_est)
% Relative error between x_ref (reference) and x_est,
% up to circular shifts, in 2-norm.
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    x_est = align_to_reference(x_est, x_ref);
    relerr = norm(x_ref-x_est, 2) / norm(x_ref, 2);

end
