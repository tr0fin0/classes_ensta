function [ u ] = BicycleToPathControl( xTrue, Path, K_rho, K_alpha, rho_threshold )
%Computes a control to follow a path for bicycle
%   xTrue is the robot current pose : [ x y theta ]'
%   Path is set of points defining the path : [ x1 x2 ... ;
%                                               y1 y2 ...]
%   u is the control : [v phi]'

    x = xTrue(1);
    y = xTrue(2);
    theta = xTrue(3);

    % initialize persistent
    persistent CURRENT_VERTICE;
    persistent CURRENT_POINT;
    persistent PATH_POINTS;

    if isempty(CURRENT_VERTICE)
        CURRENT_VERTICE = 1;
    endif

    if CURRENT_VERTICE == size(Path, 2) || isempty(PATH_POINTS)
        CURRENT_VERTICE = 1;

        x_points = linspace(Path(1, CURRENT_VERTICE), Path(1, CURRENT_VERTICE + 1), 20);
        y_points = linspace(Path(2, CURRENT_VERTICE), Path(2, CURRENT_VERTICE + 1), 20);

        PATH_POINTS = [[ x_points; y_points ]];
    endif

    if isempty(CURRENT_POINT)
        CURRENT_POINT = 1;
    endif

    % update target points
    if CURRENT_POINT <= size(PATH_POINTS, 2)
        current_target = PATH_POINTS(:, CURRENT_POINT);
        x_g = current_target(1);
        y_g = current_target(2);

        % reached current target?
        if sqrt((x - x_g)^2 + (y - y_g)^2) < rho_threshold
            CURRENT_POINT = CURRENT_POINT + 1;
        endif
    else
        if (CURRENT_VERTICE + 1) == size(Path, 2)
            x_g = Path(1, end);
            y_g = Path(2, end);
        else
            CURRENT_POINT = 1;
            CURRENT_VERTICE = CURRENT_VERTICE + 1;

            x_points = linspace(Path(1, CURRENT_VERTICE), Path(1, CURRENT_VERTICE + 1), 20);
            y_points = linspace(Path(2, CURRENT_VERTICE), Path(2, CURRENT_VERTICE + 1), 20);

            PATH_POINTS = [[ x_points; y_points ]];

            current_target = PATH_POINTS(:, CURRENT_POINT);

            x_g = current_target(1);
            y_g = current_target(2);
        endif
    endif

    % command
    rho = sqrt((y_g - y)^2 + (x_g - x)^2);
    alpha = AngleWrap( atan2(y_g - y, x_g - x) - theta );

    v = K_rho * rho;
    phi = K_alpha * alpha;

    u = [ v phi ];
end
