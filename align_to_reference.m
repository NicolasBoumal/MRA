function x_aligned = align_to_reference(x, xref)
% Given two column vectors x and xref, returns x after circularly shifting
% it such that it is optimally aligned with xref in 2-norm.
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    assert(all(size(x) == size(xref)), 'x and xref must have identical size');
    x = x(:);
    xref = xref(:);

    x_fft = fft(x);
    xref_fft = fft(xref);
    
    correlation_x_xref = real(ifft(conj(x_fft) .* xref_fft));
    [~, ind] = max(correlation_x_xref);
    x_aligned = circshift(x, ind-1);

end
