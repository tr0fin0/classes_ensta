function [ u ] = BicycleToPoseControl( xTrue,xGoal )
%Computes a control to reach a pose for bicycle
%   xTrue is the robot current pose : [ x y theta ]'
%   xGoal is the goal point
%   u is the control : [v phi]'

  x = xTrue(1);
  y = xTrue(2);
  theta = xTrue(3);

  x_delta = xGoal(1) - x;
  y_delta = xGoal(2) - y;


  rho = sqrt( (x_delta)^2 + (y_delta)^2 );
  alpha = AngleWrap( atan2(y_delta, x_delta) - theta );
  betta = AngleWrap( xGoal(3) - atan2(y_delta, x_delta) );

  K_rho = +25.0;
  K_alpha = +10.0;
  K_betta = -05.0;


  v = K_rho * rho;
  phi = K_alpha * alpha + K_betta * betta;

  u = [v, phi];

end

