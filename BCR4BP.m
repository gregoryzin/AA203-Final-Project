function [dYdt] = BCR4BP(t,Y,BCP1)
%This function computes the state derivatives of the major bicircular
% problem. The largest body is treated as the tertiary body orbiting the
% the barycenter of the two smaller bodies.
%   Detailed explanation goes here

% Author: Gregory Zin 4/21/24 

% Updates
%   4/25/24 - added STM calculation with getJacobian4BP.m
%   5/28/24 - grouped characteristic parameters mu, m3, a3, n3 into a
%             structure variable called BCP1

% extract characteristic units
mu = BCP1.mu;
a3 = BCP1.a3;
mu3 = BCP1.mu3;
n3 = BCP1.n3; 

% extract states
rvec = Y(1:3); % position vector
vvec = Y(4:6); % velocity vector
theta = Y(7);  % angle of the tertiary body in the rotating frame [rad] 

x = rvec(1);
y = rvec(2);
z = rvec(3);

% check if STM included
if length(Y) == 56 
    STM = true;
    phi = reshape(Y(8:56),7,7);
else
    STM = false;
end

% angle of the tertiary body in the rotating frame [rad]
% theta = theta0 + (n3-1)*t; 
thetadot = n3 - 1;

% distance from primary and secondary body
r1 = sqrt((x+mu)^2 + y^2 + z^2);
r2 = sqrt((x-1+mu)^2 + y^2 + z^2);

% distance from tertiary body
r3 = sqrt((x-a3*cos(theta))^2 + (y-a3*sin(theta))^2 + z^2);

% CR3BP equations of motion
rdotvec = vvec;
vdotvec = [2*rdotvec(2) + rvec(1) - (1-mu)*(rvec(1)+mu)/r1^3 - mu*(rvec(1)-1+mu)/r2^3;
    -2*rdotvec(1) + rvec(2) - (1-mu)*rvec(2)/r1^3 - mu*rvec(2)/r2^3;
    -1*(1-mu)*rvec(3)/r1^3 - mu*rvec(3)/r2^3];

% additional terms from tertiary body
vdotvec = vdotvec + [-mu3*(x-a3*cos(theta))/r3^3 - mu3*cos(theta)/a3^2; 
                    -mu3*(y-a3*sin(theta))/r3^3 - mu3*sin(theta)/a3^2; 
                    -mu3*z/r3^3];

if STM
    % Jacobian of BCR4BP1
    F = getJacobian4BP([rvec;theta],mu,mu3,a3);
    phidot = F*phi;

    dYdt = [rdotvec; vdotvec; thetadot; phidot(:)];
else
    dYdt = [rdotvec; vdotvec; thetadot];
end

end