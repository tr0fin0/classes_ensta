function [K, S, CLP] = Q15(A, B, Q, R)        
    [K, S, CLP] = lqr(A, B, Q, R);
end