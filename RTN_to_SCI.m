function R = RTN2SCI(R_SCI,T_SCI,N_SCI)
%This function returns a rotation matrix for a vector in the RTN frame 
% to the Sun centered inertial (SCI) frame.
%   Inputs:
%   R - radial unit vector defined in the SCI frame
%   T - tangential unit vector defined in the SCI frame
%   N - normal unit vector defined in the SCI frame

% Author: Gregory Zin 5/28/2024

% transformation matrix is just the column unit vectors 
R = [R_SCI/norm(R_SCI), T_SCI/norm(T_SCI), N_SCI/norm(N_SCI)];

% Example usage:
% let vector n = [-1; 0; 0] in RTN frame (anti radial direction)
% then transform to SCI frame by:
%   n_SCI = RTNtoSCI(R_SCI,T_SCI,N_SCI) * n 
%         = -1 * R_SCI/norm(R_SCI)

end

