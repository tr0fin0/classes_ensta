function [ u ] = BicycleToPathControl( xTrue, Path )
%Computes a control to follow a path for bicycle
%   xTrue is the robot current pose : [ x y theta ]'
%   Path is set of points defining the path : [ x1 x2 ... ;
%                                               y1 y2 ...]
%   u is the control : [v phi]'

    x = xTrue(1);
    y = xTrue(2);
    theta = xTrue(3);


    distances = [];
    for point = Path
      x_path = point(1);
      y_path = point(2);

      distance = sqrt( ( x_path - x )^2 + ( y_path - y )^2 );
      distances = [distances, distance];
    endfor

    [~, point_index] = min(distances);


    point_index = min(length(distances), point_index+1);


    x_delta = Path(1, point_index) - x;
    y_delta = Path(2, point_index) - y;

    rho = sqrt( (x_delta)^2 + (y_delta)^2 );
    alpha = AngleWrap( atan2(y_delta, x_delta) - theta );

    rho_reference = 0.5;
    K_rho = 25;
    K_alpha = 15;

    if rho > rho_reference
      v = K_rho * rho;
      phi = K_alpha * alpha;
    else
      v = 10;
      phi = 0;
    endif

    u = [ v phi ];

end
