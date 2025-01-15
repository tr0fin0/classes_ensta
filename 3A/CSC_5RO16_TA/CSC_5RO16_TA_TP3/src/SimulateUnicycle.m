function [xTrue,u]=SimulateUnicycle(xTrue,u)

dt=.001; % integration time
dVMax=.01; % limit linear acceleration
dWMax=.01; % limit angular acceleration

persistent v;
persistent w;

if isempty(v) v=0; end
if isempty(w) w=0; end

%limit acceleration
delta_v= min(u(1)-v,dVMax);
delta_v= max(delta_v,-dVMax);
delta_w= min(u(2)-w,dWMax);
delta_w= max(delta_w,-dWMax);

v=v+delta_v;
w=w+delta_w;

% limit control
v=min(1,v);
v=max(-1,v);
w=min(pi,w);
w=max(-pi,w);
u=[v,w];

xTrue = [xTrue(1)+v*dt*cos(xTrue(3)); xTrue(2)+v*dt*sin(xTrue(3)); xTrue(3)+w*dt];
xTrue(3) = AngleWrap(xTrue(3));
