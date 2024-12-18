function [xTrue,u]=SimulateBicycle(xTrue,u)
dt=.001; % integration time
dVMax=.01; % limit linear acceleration
dPhiMax=.008; % limit wheel speed

persistent v;
persistent phi;

if isempty(v) v=0; end
if isempty(phi) phi=0; end

%limit acceleration
delta_v= min(u(1)-v,dVMax);
delta_v= max(delta_v,-dVMax);
delta_phi= min(u(2)-phi,dPhiMax);
delta_phi= max(delta_phi,-dPhiMax);

v=v+delta_v;
phi=phi+delta_phi;

% limit control
v=min(1,v);
v=max(-1,v);
phi=min(1.2,phi);
phi=max(-1.2,phi);

xTrue = [xTrue(1)+v*dt*cos(xTrue(3)); xTrue(2)+v*dt*sin(xTrue(3)); xTrue(3)+v/0.5*dt*tan(phi)];
xTrue(3) = AngleWrap(xTrue(3));
u=[v,phi];