function recursiveSquares(input)


    N       = input.N;
    X       = input.X;
    Z       = input.Z;
    sigmab  = input.sigmab;


    for n = 0:N-1
        m_np = 0;
        R_np = sigmab;
        H_np = [
            cos(om1 * (n+1) * Te),
            cos(om2 * (n+1) * Te),
            cos(om3 * (n+1) * Te)
        ];

        S_np = H_np * P_n * H_np' + R_np;
        K_np = P_n * H_np' * S_np;
        theta = theta_n + K_np * (Z(n+1) - m_np - H_np * theta_n);
        P_np = P_n - K_np * H_np * P_n;
        
    end

    return theta
end