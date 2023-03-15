function OBS = Q19(A, C)
    OBS = [];
    for i = 0:size(A, 1)-1
        OBS = [OBS; (C * (A^i))];
    end
    
    if rank(OBS) ~= 4
        disp(['warning: system not observable, rank ' num2str(rank(OBS))]);
    else
        disp(['success: system observable, rank ' num2str(rank(OBS))]);
    end
end