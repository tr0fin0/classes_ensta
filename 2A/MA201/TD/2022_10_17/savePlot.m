function savePlot(saveName)

    fileType = 'png';
    pPath = 'C:\Users\Admin-PC\Documents\git_repository\classes_ensta\2A\MA201\TD\10_10_2022\images';

    saveas(gca, fullfile(pPath, saveName), fileType);
end