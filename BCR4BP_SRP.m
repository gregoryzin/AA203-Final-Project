function [dYdt] = BCR4BP_SRP(t,Y,U,BCP1,BCP2,SC)
%This function computes the state derivatives of the major bicircular
% problem with solar radiation pressure. The Sun is treated as 
% the tertiary body orbiting the barycenter of the two smaller bodies.
%   Detailed explanation goes here

% Author: Gregory Zin 5/28/24 

% extract parameters of spacecraft model
m = SC.mass;
A = SC.area;
R_diff = SC.R_diff; 
R_spec = SC.R_spec;

% extract characteristic units
mu = BCP1.mu;
LU = BCP1.LU;
TU = BCP1.TU;
a3 = BCP1.a3;
mu3 = BCP1.mu3;
n3 = BCP1.n3; 

% extract control inputs
alpha = U(1);
beta = U(2);

% extract states 
rvec = Y(1:3); % position vector
vvec = Y(4:6); % velocity vector
theta = Y(7);  % angle of the tertiary body in the rotating frame [rad] 

x = rvec(1);
y = rvec(2);
z = rvec(3);

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

% acceleration terms from tertiary body
vdotvec = vdotvec + [-mu3*(x-a3*cos(theta))/r3^3 - mu3*cos(theta)/a3^2; 
                    -mu3*(y-a3*sin(theta))/r3^3 - mu3*sin(theta)/a3^2; 
                    -mu3*z/r3^3];


% convert current state from BCP1 -> BCP2 
[t2,Y2] = BCP1toBCP2(t,Y',BCP1,BCP2);

% convert current state from BCP2 -> SCI frame 
[t_SCI,Y_SCI] = rot2inert(t2,Y2,BCP2.mu,BCP2.LU,BCP2.TU,1);

% get radial, tangential, normal vectors in SCI frame
R_SCI = Y_SCI(1:3)';
N_SCI = cross(Y_SCI(1:3)',Y_SCI(4:6)');
T_SCI = cross(N_SCI,R_SCI);

% flat plate unit normal vector in RTN frame
%   alpha - defines azimuth in the RTN frame [rad]
%   beta - defines elevation angle with respect RTN frame [rad]
% When alpha = beta = 0, the solar sail normal vector points along the
% -R direction in the RTN frame.
n_RTN = [cos(alpha+pi)*cos(beta); sin(alpha+pi)*cos(beta); sin(beta)];

% convert the normal vector from RTN to SCI frame
n_SCI = RTN_to_SCI(R_SCI, T_SCI, N_SCI) * n_RTN;

% cosine of angle between -R_SCI and n_SCI [rad]
% we take the negative of R_SCI to get position vector from spacecraft
% pointing to the Sun
cos0 = dot(-R_SCI/norm(R_SCI),n_SCI);

% solar radiation pressure [N/m^2]
P = SRP(norm(R_SCI)); 

% SRP force vector in SCI frame [N]
F_SRP = -P * A * (2*(R_diff/3 + R_spec*cos0)*n_SCI ...
    + (1-R_spec)*-R_SCI/norm(R_SCI)) * max(cos0,0);

% acceleration due to SRP in SCI frame [m/s^2]
a_SCI = F_SRP/m; 

% transform the acceleration vector from SCI -> BCP2 frame
[t2,a2] = SCI_to_BCP2(t_SCI,a_SCI,BCP2);

% transform the acceleration vector from BCP2 -> BCP1 frame
[t1,a1] = BCP2_to_BCP1(t2,a2,pi-theta,BCP1,BCP2);

% nondimensionalize the acceleration and add to vdotvec  
vdotvec = vdotvec + a1 * TU^2/(LU*1000);

% combine state derivative
dYdt = [rdotvec; vdotvec; thetadot];

end