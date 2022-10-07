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
    theta_0 = zeros(3,1);
    P_0 = 1000 * eye(3);

    % initialization
    theta_n = theta_0;
    P_n = P_0;

    % algorithm
    for n = 0:N-1
        % 
        m_np1 = 0;
        R_np1 = sigmab;
        H_np1 = [cos(om1*(n+1)*Te), cos(om2*(n+1)*Te), cos(om3*(n+1)*Te)];

        % recursive
        size(H_np1)
        size(P_n)
        size(R_np1)
        size(S_np1)
        S_np1       = H_np1 * P_n * H_np1' + R_np1;
        K_np1       = P_n * H_np1 / S_np1;
        theta_np1   = theta_n + K_np1 * (Z(n+1) - m_np1 - H_np1 * theta_n);
        P_np1       = P_n - K_np1 * H_np1 * P_n;

        % 
        theta_n = theta_np1;
        P_n = P_np1;
    end

    theta = theta_n;
end