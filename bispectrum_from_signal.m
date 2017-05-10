function B = bispectrum_from_signal(s)
% Given a signal s of length N, returns its bispectrum B of size NxN.
% The convention is as follows:
%
%   shat = fft(s),
%
%   B(f1, f2) = shat(f1) conj(shat(f2)) shat(f2-f1),
%
% where indexing f1, f2 is from 0 to N-1 and understood modulo N.
%
% In Matlab's notation, indexing is from 1 to N, so that B(1, 1)
% corresponds to frequencies f1 = f2 = 0.
%
% Note: if s is real, B is Hermitian.
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    s = s(:);
    
    shat = fft(s);
    
    % The term y(f2-f1) has a circulant structure owing to the signal being
    % discrete, so that its discrete Fourier transform is thought of as
    % periodic. If s is real, C is Hermitian.
    C = circulant(shat);
    
    % Produce the bispectrum.
    B = (shat*shat') .* C;

end
