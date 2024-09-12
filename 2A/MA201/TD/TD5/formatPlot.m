function formatPlot(titleText, legendText, xLabel, yLabel)
    pSize = [0 0 18 18];
    legendLocation = "Best";

    if not(isempty(titleText))
        title(titleText, 'Interpreter','tex')
    end

    if not(isempty(legendText))
        legend(legendText, "location", legendLocation, 'Interpreter', 'tex')
    end

    if not(isempty(xLabel))
        xlabel(xLabel);
    end

    if not(isempty(yLabel))
        ylabel(yLabel);
    end

    grid on;
    set(gcf, 'PaperPosition', pSize);
end