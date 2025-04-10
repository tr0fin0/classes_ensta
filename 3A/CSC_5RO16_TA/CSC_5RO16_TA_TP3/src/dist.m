function [ error ] = dist( xTrue,xGoal )
%DIST Summary of this function goes here
%   Detailed explanation goes here
error=xGoal-xTrue;
if (size(error,1)==3)
    error(3)=AngleWrap(error(3));
end
end

