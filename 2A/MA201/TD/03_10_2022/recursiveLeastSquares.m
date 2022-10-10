function theta = recursiveLeastSquares(input)
    % ========================================
    % Recursive Least Squares
    % ========================================


    % initial variables
    N       = input.N;
    Te      = input.Te;
    Z       = input.Z;
    om1     = input.om1;
    om2     = input.om2;
    om3     = input.om3;
    sigmab  = input.sigmab;


    % initial conditions
    theta = zeros(3,1);
    P = eye(3);
    R = sigmab * eye(N);

    % setup
    H  = [
        cos(om1 * Te * (1:N));
        cos(om2 * Te * (1:N));
        cos(om3 * Te * (1:N));
    ]';
    theory = (H'*H) \ (H'*Z);

    % recursion
    for i = 1:N
        [P, theta] = rec(P, theta, H, Z, R);
    end
    theory

    function [P, theta] = rec(P, theta, H, Z, R)
        S = H * P * H' + R;
        K = (P * H') / S;
        theta = theta + K * (Z - H*theta);
        P = P - K*H*P;
    end
end