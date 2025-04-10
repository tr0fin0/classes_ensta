function [ u ] = BicycleToPoseControl( xTrue, xGoal, K_rho, K_alpha, K_beta )
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
  beta = AngleWrap( xGoal(3) - atan2(y_delta, x_delta) );

  v = K_rho * rho;
  phi = K_alpha * alpha + K_beta * beta;

  u = [v, phi];
end
