function X = kalmanFilterSimple(data, x, F, B, H, Q, R, u)
    % ========================================
    % Kalman Filter Calculation
    % ========================================
    % data : vector of information to be filter
    % x    : Vector of Initial Guess


    % considering a system model given by the following equation:
    %   | x_{k+1} = F x_{k} + B u_{k} + w_{k}
    %   | z_{k}   = H x_{k} +   v_{k}
    % 
    % where:
    %   F   : Matrix of Transition  (n x n)
    %   B   : Matrix of Control     (n x n)
    %   H   : Matrix of Observation (m x n)
    %   x   : Vector of States      (n x 1)
    %   u   : Vector of Entrates    (n x 1)
    %   z   : Vector of Mesures     (m x 1)
    %   where:
    %       n   : number of states
    %       m   : number of mesured states
    
    %   w_{k} : noise of model  given by: N(0, W)
    %       Q_{k} Covariance Matrix (n x n)
    % 
    %   v_{k} : noise of mesure given by: N(0, V)
    %       R_{k} Covariance Matrix (m x m)
    % 
    %           N(x,y) : Normal Distribution

    % theoretically F, B, H could vary over k interactions but
    % in this algorithm we consider as constant


    N   = length(data);     % size of observation
    X   = [];               % matrix to return values


    % Matrix Identity
    I  = eye(length(F));

    % support matrixes
    P  = 1000 * I;          % initial estimation
    z  = data;


    for i = 1 : N
        % setup
        zk = z(:,i);
        uk = u(:,i);

        % Prediction
        x = F * x + B  * uk;    % (n x 1)
        P = F * P * F' + Q;     % (n x n)


        % Correction
        y = zk - H * x;         % (m x 1)
        S = H  * P * H' + R;    % (m x m)
        K = P  * H' * inv(S);   % (n x m)


        % Update
        P = (I - K * H) * P;    % (n x n)
        x = x + K * y;          % (n x 1)


        % Saving
        X = [X, x];             % (n x N)
    end

end