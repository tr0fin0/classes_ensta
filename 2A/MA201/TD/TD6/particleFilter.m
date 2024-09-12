function [x, t] = particleFilter(Nr, N, Z, F, H, W, V)

    % ========================================
    % Particules Filter
    % ========================================
    % x : vector of values
    % t : vector of trace


    % inicialization
    x  = [];
    t  = [];

    x0 = [0; 0; 0; 0];
    P0 = 1/Nr * eye(4);

    Pk = P0;
    xk = x0;
    % TODO automatic sizing
    
    
    % particles generation
    size(F)
    size(Pk)
    size(sqrtm(Pk))
    size(Nr)
    size(randn(Nr))
    xki = F + randn(Nr) .* sqrtm(Pk)
    
    % error calculation
    eki = zk - H
    
    % vraisemblance calculation
    Pp = 1/(Nr^2) * P0;
    M = H * Pp * H' + V
    qi = (1/(sqrt(2*pi))) * exp(-(1/2)*(eki' * inv(M) * eki)) % pi ^ n where n is ??
    
    for k = 1 : N
        for i = 1 : Nr
            xki = xk + sqrt(Pk) * randn(4, 1);
            xki = F + sqrt(W) * randn(4, 1);

            ek = zk - hk;
            qi = eki; % p*(eki)
            wki = qi / sum(qi)
        end

        xk = sum(wki * xki);
        Pk = sum(wki * (xk - xki) * (xk - xki)');
    end
    % x     % debuging
    % t     % debuging
end