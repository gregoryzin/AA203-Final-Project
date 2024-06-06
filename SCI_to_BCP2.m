function [t2,V2] = SCI_to_BCP2(T,V,BCP2,phi)
%This function converts an arbitrary vector defined in the Sun centered
% inertial (SCI) frame to a vector in the rotating BCP2 frame. The BCP2
% frame is centered at B2, the Sun-B1 barycenter. Functionally, this
% performs similarly to 'inert2rot.m', but the vector is irrespective of 
% the primary or secondary bodies. 

% Input:
%   V - vector defined in the SCI frame at time T
%   BCP2 - structure variable containing mu, LU, TU for BCP2
%   phi - initial offset angle of rotating frame wrt to inertial frame [rad]

% Output:
%   t2 - nondimensionalized time in the BCP2
%   V2 - the vector V in the BCP2 rotating frame

% Author: Gregory Zin 5/28/2024

% Changelog: 
%   6/5/2024 - fixed rotation matrix to use T instead of t2

if nargin < 4
    phi = 0;
end

% extract characteristic parameters
mu = BCP2.mu;
LU = BCP2.LU;
TU = BCP2.TU;

% nondimensionalized time [non]
t2 = T/TU;

% mean motion from the period [rad/sec]
n = 1/TU;
w = [0; 0; 1]; % angular velocity vector

% let X, Y, Z be axes of the rotating frame defined in the SCI
% X = [cos(n*T+phi); sin(n*T+phi); 0];
% Y = [-sin(n*T+phi); cos(n*T+phi); 0];
% Z = [0; 0; 1];

% let R = rotation matrix from rotating to SCI frame
R = [cos(n*T+phi), -sin(n*T+phi), 0;
    sin(n*T+phi), cos(n*T+phi), 0;
    0, 0, 1];

% use transpose of R to convert from inertial to rotating frame
V2 = R' * V;

end

