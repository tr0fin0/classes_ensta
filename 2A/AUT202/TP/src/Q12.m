function K = Q12(A, B, w)
    P = [-w; -2*w; -w+1i*w; -w-1i*w];

    K = place(A, B, P);
    
    diff = sort(P) - sort(eig(A-B*K));
    error = false;
    for i = 1:size(diff,1)
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