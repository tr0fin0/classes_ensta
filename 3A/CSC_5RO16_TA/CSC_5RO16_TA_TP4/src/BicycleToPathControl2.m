function [ u ] = BicycleToPathControl2( xTrue, Path, window_size )
    %Computes a control to follow a path for bicycle
    %   xTrue is the robot current pose : [ x y theta ]'
    %   Path is set of points defining the path : [ x1 x2 ...
    %                                               y1 y2 ...]
    %   u is the control : [v phi]'
    %   window_size is the antecipation window:
    %       1 : anticipe,
    %       5 : anticipe bien, controle plus souple,
    %      20 : coupe un peu, controle très souple,
    %     100 : coupe un peu,
    %    1000 : triche !

    persistent goalWaypointId xGoal;


    if xTrue == [0;0;0]
        goalWaypointId = 1;
        xGoal = Path(:,1);
    end

    rho = 0.3;
    dt = 0.01;


    vmax = 2.0;
    dmax = vmax * dt;

    list_points = [];
    xtemp = xTrue;

    while size(list_points, 2) < window_size
        if norm((Path(:, goalWaypointId) - xtemp)(1:2)) < rho
            xtemp = Path(:, goalWaypointId);
            list_points = [list_points, xtemp];

            goalWaypointId = goalWaypointId + 1;
            goalWaypointId = min(goalWaypointId, size(Path, 2));
        else
            direction = Path(:, goalWaypointId) - xtemp;
            direction = direction / norm(direction);
            xtemp = xtemp + dmax * direction;
            list_points = [list_points, xtemp];
        endif
    end


    anticipation = window_size;

    Krho = 10;
    Kalpha = 5;

    error = list_points(:, anticipation) - xTrue;
    goalDist = norm(error(1:2));
    AngleToGoal = AngleWrap(atan2(error(2), error(1)) - xTrue(3));

    u(1) = Krho * goalDist/(window_size * 10);
    u(2) = Kalpha * AngleToGoal;
end
