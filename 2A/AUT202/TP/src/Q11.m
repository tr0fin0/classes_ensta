function COM = Q11(A, B)
    COM = B;
    for i = 1:size(A, 1)-1
        COM = [COM, (A^i) * B];
    end
    
    if rank(COM) ~= 4
        disp(['warning: system not controable, rank ' num2str(rank(COM))]);
    else
        disp(['success: system controable, rank ' num2str(rank(COM))]);
    end
end