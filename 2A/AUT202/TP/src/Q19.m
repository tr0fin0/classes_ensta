function OBS = Q19(A, C)
    OBS = C;
    for i = 1:size(A, 1)-1
        OBS = [C; C * (A^i)];
    end
    
    if rank(OBS) ~= 4
        disp(['warning: system not observable, rank ' num2str(rank(OBS))]);
    else
        disp(['success: system observable, rank ' num2str(rank(OBS))]);
    end
end