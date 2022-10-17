function savePlot(saveName)

    fileType = 'png';
    pPath = 'C:\Users\Admin-PC\Documents\git_repository\classes_ensta\2A\MA201\TD\2022_10_10\images';

    saveas(gca, fullfile(pPath, saveName), fileType);
end