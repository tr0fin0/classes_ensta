function [x, t] = kalmanFilter(N, Z, F, H, W, V)
    % ========================================
    % Kalman Filter
    % ========================================
    % x : vector of values
    % t : vector of trace



    % inicialization
    x  = [];
    t  = [];

    xk = [0; 0; 0; 0];
    Pk = 1000 * eye(4);
    % TODO automatic sizing


    for i = 1 : N
        % prediction
        xk = F * xk;
        Pk = F * Pk * F' + W;

        % correction
        K  = Pk * H' * inv(H * Pk * H' + V);

        % update
        xk = xk + K * (Z(:,i) - H * xk);
        Pk = (eye(size(K*H)) - K * H) * Pk;

        % saving
        x = [x, xk];
        t = [t, trace(Pk)];
    end
    % x     % debuging
    % t     % debuging
end