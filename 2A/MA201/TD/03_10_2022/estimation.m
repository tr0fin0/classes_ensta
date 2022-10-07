function [N, Z] = estimation(func, input)

    % initial variables from func data
    N   = input.N;      % number of samples
    Z   = input.Z;      % value  of samples
    w1  = input.om1;    % value  of samples
    w2  = input.om2;    % value  of samples
    w3  = input.om3;    % value  of samples
    Te  = input.Te;     % constant


    inputName = inputname(1);

    if inputName == 'GM'
        titleName = 'Gauss Markov';

    elseif inputName == 'RLS'
        titleName = 'Recursive Least Squares';

    else
        error('undefined function name')
    end


    % theta constants
    theta = func(input);
    a = theta(1)
    b = theta(2)
    c = theta(3)

    % x axis values
    % N = Te * (1:0.001:N)';
    N = (1 : 0.001 : N)';

    % y axis values (estimation)
    Z = a*cos(w1 * N) + b*cos(w2 * N) + c*cos(w3 * N);

end