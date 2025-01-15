function [ u ] = UnicycleToPoseControl( xTrue, xGoal, alpha_maximum, K_rho, K_alpha, K_beta )
  %Computes a control to reach a pose for unicycle
  %   xTrue is the robot current pose : [ x y theta ]'
  %   xGoal is the goal point
  %   u is the control : [v omega]'

  x = xTrue(1);
  y = xTrue(2);
  theta = xTrue(3);

  x_delta = xGoal(1) - x;
  y_delta = xGoal(2) - y;

  rho = sqrt( (x_delta)^2 + (y_delta)^2 );
  alpha = AngleWrap( atan2(y_delta, x_delta) - theta );

  v = K_rho * rho;

  if abs(alpha) > alpha_maximum
    v = 0;
  endif

  if rho > 0.05
    omega = K_alpha * alpha;
  else
    omega = K_beta * AngleWrap( xGoal(3) - theta );
  endif

  u = [v, omega];
end
