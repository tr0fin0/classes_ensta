function [x, t] = kalmanFilter(N, Z, F, H, W, V)
    % ========================================
    % Kalman Filter
    % ========================================


    % inicialization
    x  = [];
    xk = [0; 0; 0; 0];
    Pk = 1000 * eye(4);


    for i = 1 : N
        % prediction
        xk = F * xk;
        Pk = F * Pk * F' + W;

        % correction
        K  = Pk * H' * inv(H * Pk * H' + V);

        % update
        xk = xk + K * (Z(:,i) - H * xk);
        Pk = (eye(size(K*H)) - K * H) * Pk;

        x = [x, xk];
    end
    % x     % debuging
end