function [t1,V1] = BCP2_to_BCP1(t2,V2,theta2,BCP1,BCP2,OE)
%This function converts an arbitrary vector defined in the BCP2 frame to a 
% vector in the rotating BCP1 frame. The BCP2 frame is centered at B2, the 
% Sun-B1 barycenter. The BCP1 frame is centered at B1, the Earth-Moon
% barycenter. 

% Inputs:
%   V2 - arbitrary vector in BCP2 frame at time t2
%   BCP1, BCP2 - structure variables containing parameters for the two
%       frames. BCP1 contains the parameters of the BCR4BP (aka major BCP).
%       BCP2 contains parameters of the minor BCP, or just the CR3BP for
%       the Sun-EM problem. 
%   theta2 - Moon angle in BCP2

% Author: Gregory Zin 5/28/2024

% extract characteristic lengths and times
mu1 = BCP1.mu;
LU1 = BCP1.LU;
TU1 = BCP1.TU;

mu2 = BCP2.mu;
LU2 = BCP2.LU;
TU2 = BCP2.TU;

% check if orbital elements of Sun are specified
if nargin < 6
    Omega = 0;  % right ascension of ascending node
    i = 0;    % inclination of the Sun orbital plane wrt Earth-Moon plane
else
    Omega = OE(1);
    i = OE(2);
end

% transform time t2 to dimensional units
T = t2 .* TU2;

% transform T into nondimensional BCP1 time units
t1 = T ./ TU1;

% rotation about z-axis (2.56)
C1 = [cos(theta2 - Omega), -sin(theta2 - Omega), 0;
    sin(theta2 - Omega), cos(theta2 - Omega), 0;
    0, 0, 1];

% rotation about intermediate y-axis (2.57)
C2 = [cos(i), 0, sin(i);
    0, 1, 0;
    -sin(i), 0, cos(i)];

% rotation about intermediate z-axis (2.58)
C3 = [cos(Omega), -sin(Omega), 0;
    sin(Omega), cos(Omega), 0;
    0, 0, 1];

% the rotation C = C1 * C2 * C3 takes a vector from BCP1 -> BCP2
C = C1 * C2 * C3;

% we want to go the other direction, so we use the transpose
% C' to go from BCP2 -> BCP1
V1 = C' * V2;

end

