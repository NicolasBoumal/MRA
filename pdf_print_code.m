function pdf_print_code(h, filename, fontsz)
% Print a matlab figure to a PDF.
%
% Inputs:
%   h -- Figure handle (e.g., gcf)
%   filename -- output file name (e.g., 'myfigure.pdf')
%   fontsz -- font size for all axes, labels, ... (optional)
%
% Nicolas Boumal, Feb. 21, 2017
    
    if ~exist('fontsz', 'var')
        fontsz = [];
    end

    % Reducing paper size to the figure's needs
    set(h, 'Units', 'Inches');
    pos = get(h, 'Position');
    set(h, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);

    % This block entirely to set font size
    if ~isempty(fontsz)
        all_axes = findall(gcf, 'Type', 'axes');
        for haxes = all_axes
            set([haxes ; findall(haxes, 'Type', 'text')], 'FontSize', fontsz);
        end
        set(findall(gcf, 'Type', 'text'), 'FontSize', fontsz);
        drawnow;
    end

    % Print to PDF
    print(h, filename, '-dpdf', '-r0'); % can also try '-bestfit'

    % Hopefully, pdfcrop is installed so we can get rid of extra white space
    margins = 1;
    system(sprintf('pdfcrop -margins %d %s %s', margins, filename, filename));
    
end
