function L = Q21(A, C, w)
    P = [-w; -3*w; -2*w+1i*w; -2*w-1i*w];

    L = place(A', C', P)';
    
    diff = sort(P) - sort(eig(A-L*C));
    error = false;
    for i = 1:size(P,1)
        if diff(i) > 1e-9
            error = true;
        end
    end
    if error
        disp('error: fail to perform closed-loop pole assignment');
    else
        disp('success: closed-loop pole assignment');
    end
end