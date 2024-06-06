function P = SRP(r)
%This function computes the solar radiation pressure at given distance r
%from the Sun.
% Input:
%   r - distance from the Sun in km
% Output:
%   P - solar pressure in Pascals (N/m^2)

% constants
L = 3.828e26; % luminosity of the Sun [W]
c = 299792458; % speed of light [m/s]

% solar flux [W/m^2]
F = L/(4*pi*(1000*r)^2);

% solar pressure [N/m^2]
P = F/c;

end