function [I1, I2, A, B, C, K] = P3(x10, x20, x30, x40, w)
    % initial variables
    I1 = [x10, x20, x30, x40];
    I2 = [0,0,0,0];
    
    % matrixes
    [J, m, sig, g] = Q5();
    A   = [...
        0       1 0             0;...
        0       0 -(g/(1+sig))  0;...
        0       0 0             1;...
        -m*g/J  0 0             0;
    ];

    B   = [
        0;
        0;
        0;
        1/J;
    ];

    C   = [
        1 0 0 0;
        0 0 1 0;
    ];

    
    if w == inf
        Q = eye(4); 
        R = 1;
        [K, S, CLP] = Q15(A, B, Q, R);
    else
        K = Q12(A, B, w);
    end    
end