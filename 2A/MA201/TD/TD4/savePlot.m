function savePlot(titleText, legendText, saveName, xLabel, yLabel)
    fileType = 'png';
    legendLocation = "southeast";
    pPath = 'C:\Users\Admin-PC\Documents\git_repository\classes_ensta\2A\MA201\TD\2022_10_03\images';
    pSize = [0 0 18 18];


    if not(isempty(titleText))
        title(titleText,  'Interpreter','tex')
    end

    if not(isempty(legendText))
        legend(legendText, "location", legendLocation, 'Interpreter', 'tex')
    end

    if not(isempty(xLabel))
        xlabel(xLabel)
    end

    if not(isempty(yLabel))
        ylabel(yLabel);
    end

    grid on;
    set(gcf, 'PaperPosition', pSize); 
    saveas(gca, fullfile(pPath, saveName), fileType);
end