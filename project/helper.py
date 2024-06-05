"""
Helper Code for AA 203 Final Project - Gregory Zin

This code contains all the dynamics, force modeling, and frame transformation functions I originally wrote in MATLAB.

"""

import jax
import jax.numpy as jnp

import numpy as np


# def CR3BP_SRP(s,u):
#     """Compute the state derivative in the CR3BP."""
#     # Sun-Earth CR3BP constants
#     # gravitational parameter [km^3 s^-2]
#     mu_E = 398600.435507 # Earth
#     mu_S = 1.32712440041279419e11 # Sun
#     mu = mu_E/(mu_E + mu_S) 

#     # position of Sun in CR3BP
#     r_Sun = jnp.array([-mu, 0, 0])

#     LU = 1.49598e8 # km
#     TU = jnp.sqrt(LU**3/(mu_E + mu_S)) # seconds

#     # spacecraft sail model (single plate)
#     m = 39000 # kg
#     A = 345 # m^2
#     R_diff = 0.5 
#     R_spec = 0.5

#     # extract states
#     x, y, z, vx, vy, vz = s
#     α, β = u
#     sinα, cosα, sinβ, cosβ = jnp.sin(α), jnp.cos(α), jnp.sin(β), jnp.cos(β)

#     # flat plate unit normal vector
#     n_B = jnp.array([cosα*cosβ, sinα*cosβ, sinβ])

#     # cosine of angle between r_s and n_B
#     cosθ = 


# gravitational parameters [km^3 s^-2]
global mu_M, mu_E, mu_S
mu_M = 4902.800118   # Moon
mu_E = 398600.435507 # Earth
mu_S = 1.32712440041279419e11 # Sun

# BCP constants
global mu, mu2, mu3, a3, n3, LU1, TU1, LU2, TU2

mu3 = mu_S/(mu_E + mu_M) 
a3 = 1.49598e8/384399.014
n3 = jnp.sqrt((1 + mu3)/a3**3)

# Earth-Moon characteristic parameters (BCP1 frame)
mu = mu_M/(mu_E + mu_M) # Earth is primary, Moon is secondary body
LU1 = 384399.014 # km
TU1 = jnp.sqrt(LU1**3/(mu_E + mu_M)) # seconds

# Sun-B1 characteristic parameters (BCP2 frame)
mu2 = (mu_E + mu_M)/(mu_E + mu_M + mu_S) # treating Sun as primary and combined Earth-Moon as secondary body
LU2 = 1.49598e8 # km
TU2 = jnp.sqrt(LU2**3/(mu_E + mu_M + mu_S)) # seconds


def BCR4BP_SRP(s,u):
    """Compute the state derivative in the BCR4BP with solar radiation pressure (SRP)."""
    # spacecraft sail model (single plate)
    m = 39000 # kg
    A = 345 # m^2
    R_diff = 0.5 
    R_spec = 0.5

    # call BCR4BP function dynamics
    ds = BCR4BP(s)

    # convert current state from BCP1 to BCP2 frame
    t2, s2 = BCP1toBCP2(s)

    # convert BCP2 state to Sun centered inertial (SCI) frame
    t_SCI, s_SCI = BCP2toSCI(t2,s2) 
    R_SCI = s_SCI[0:3]
    V_SCI = s_SCI[4:7]

    # get radial, tangential, normal vectors in SCI frame
    N_SCI = jnp.cross(R_SCI, V_SCI) # normal
    T_SCI = jnp.cross(N_SCI, R_SCI) # tangential

    # extract control variables
    α, β = u
    sinπα, cosπα, sinβ, cosβ = jnp.sin(np.pi+α), jnp.cos(np.pi+α), jnp.sin(β), jnp.cos(β)

    # flat plate unit normal vector in SCI frame
    n_SCI = jnp.array([cosπα*cosβ, sinπα*cosβ, sinβ])

    # cosine of angle between r_s and n_B
    cosθ = jnp.dot(-R_SCI/jnp.linalg.norm(R_SCI), n_SCI)

    # solar radiation pressure [N/m^2]
    P = SRP(jnp.linalg.norm(R_SCI))

    # SRP force vector in SCI frame [N]
    F_SCI = -P * A * (2*(R_diff/3 + R_spec*cosθ)*n_SCI + (1-R_spec)*-R_SCI/jnp.linalg.norm(R_SCI)) * jnp.max(jnp.array([cosθ,0]))

    # acceleration due to SRP in SCI frame [m/s^2]
    a_SCI = F_SCI/m

    # transform acceleration vector from SCI to BCP2 frame
    t2, a2 = SCItoBCP2(t_SCI,a_SCI)

    # transform acceleration vector from BCP2 to BCP1 frame
    t1, a1 = BCP2toBCP1(t2,a2)

    # nondimensionalize SRP acceleration and add to BCR4BP state derivative
    ds = jnp.array(
        [
            ds[0],
            ds[1],
            ds[2],
            ds[3] + a1[0] * TU1**2/(LU1*1000),
            ds[4] + a1[1] * TU1**2/(LU1*1000),
            ds[5] + a1[2] * TU1**2/(LU1*1000),
            ds[6]
        ]
    )
    return ds

def BCR4BP(s):
    """Compute the state derivative in the BCR4BP."""
    # call CR3BP dynamics
    ds = CR3BP(s)
    
    # extract states
    x, y, z, vx, vy, vz, θ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)

    # position of tertiary body (Sun)
    xs = a3*cosθ
    ys = a3*sinθ
    zs = 0

    # distance to tertiary body
    r3 = jnp.sqrt((x-xs)**2 + (y-ys)**2 + (z-zs)**2)

    # add acceleration to CR3BP state derivative
    ds = jnp.array(
        [
            ds[0],
            ds[1],
            ds[2],
            ds[3] - mu3*(x-a3*cosθ)/r3**3 - mu3*cosθ/a3**2,
            ds[4] - mu3*(y-a3*sinθ)/r3**3 - mu3*sinθ/a3**2,
            ds[5] - mu3*z/r3**3,
            n3 - 1
        ]
    )
    return ds


def CR3BP(s):
    """Compute the state derivative in the CR3BP."""
    # extract states
    x, y, z, vx, vy, vz = s[0:6]

    # position of primary body (Earth)
    xE = -mu

    # position of secondary body (Moon)
    xM = 1-mu

    # distance to primary and secondary body
    r1 = jnp.sqrt((x-xE)**2 + (y)**2 + (z)**2)
    r2 = jnp.sqrt((x-xM)**2 + (y)**2 + (z)**2)


    ds = jnp.array(
        [
            vx,
            vy,
            vz,
            2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3,
            -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3,
            -1*(1-mu)*z/r1**3 - mu*z/r2**3
        ]
    )
    return ds


def BCP1toBCP2(s):
    # This function converts the state s from the BCP1 frame (Earth-Moon) to the BCP2 frame (Sun-B1)

    # extract states
    x, y, z, vx, vy, vz, θ_S = s

    # position and velocity vector
    rvec1 = jnp.array([x, y, z])
    vvec1 = jnp.array([vx, vy, vz])

    # assume that time T is linearly related to θ_S by equation: θ_S = (n3-1)*T
    # and that θ_S = 0 when t = 0
    T = θ_S/(n3 - 1)

    # convert to nondimensional BCP2 time
    t2 = T/TU2

    # angle of the Moon in BCP2
    θ_M = np.pi - θ_S # rad
    sinθ_M, cosθ_M = jnp.sin(θ_M), jnp.cos(θ_M)

    # rate of Moon angle
    ω_M = 1 - n3 # rad/s

    # rotation matrix
    C = jnp.array([[cosθ_M, -sinθ_M, 0],
          [sinθ_M, cosθ_M, 0],
          [0, 0, 1]])
    
    Cdot = jnp.array([[-sinθ_M, -cosθ_M, 0],
          [cosθ_M, -sinθ_M, 0],
          [0, 0, 1]]) * ω_M
    
    # apply transformation matrices C and Cdot
    rvec2 = C @ rvec1
    vvec2 = C @ vvec1 + Cdot @ rvec1

    # convert to dimensional units
    Rvec2 = rvec2 * LU1
    Vvec2 = vvec2 * LU1/TU1 

    # convert to BCP2 nondimensional units and shift origin from B1 to B2
    s2 = jnp.array(
        [
            1 - mu2 + Rvec2[0]/LU2,
            Rvec2[1]/LU2,
            Rvec2[2]/LU2,
            Vvec2[0]*(TU2/LU2),
            Vvec2[1]*(TU2/LU2),
            Vvec2[2]*(TU2/LU2),
            θ_M
        ]
    )

    return t2, s2


def BCP2toSCI(t2,s2):
    # This function convert the state s2 from the BCP2 frame to a Sun centered inertial frame (SCI).
    
    # extract states
    x, y, z, vx, vy, vz = s2[0:6]

    # position and velocity vector
    rvec2 = jnp.array([x+mu2, y, z]) # with respect to Sun in BCP2 frame
    vvec2 = jnp.array([vx, vy, vz])

    # dimensionalized time
    T = t2 * TU2

    # mean motion is the inverse of TU2 [rad/sec]
    n = 1/TU2
    ω = jnp.array([0, 0, n]) # angular velocity vector

    # rotation matrix from rotating to inertial frame
    R = jnp.array([[jnp.cos(n*T), -jnp.sin(n*T), 0],
                   [jnp.sin(n*T), jnp.cos(n*T), 0],
                   [0, 0, 1]])
    
    # apply rotation
    rho = LU2 * R @ rvec2
    rhodot = (LU2/TU2) * R @ vvec2 + jnp.cross(ω,rho)
    
    # create jnp array of the states
    S = jnp.array(
        [
            rho[0],
            rho[1],
            rho[2],
            rhodot[0],
            rhodot[1],
            rhodot[2]
        ]
    )

    return T, S

def SRP(R):
    # This function computes the solar radiation pressure at given distance r 
    # from the Sun.
    # Input:
    #   R - distance from the Sun in km
    # Output:
    #   P - solar pressure in Pascals (N/m^2)
    
    # constants
    L = 3.828e26 # luminosity of the Sun [W]
    c = 299792458 # speed of light [m/s]

    # solar flux [W/m^2]
    F = L/(4*np.pi*(1000*R)**2)

    # solar pressure [N/m^2]
    P = F/c

    return P

def SCItoBCP2(T,a_SCI):
    # This function transforms a vector from the SCI frame to the BCP2 frame

    # nondimensionalized time
    t2 = T/TU2

    # mean motion is the inverse of TU2 [rad/sec]
    n = 1/TU2

    # rotation matrix from rotating to inertial frame
    R = jnp.array([[jnp.cos(n*T), -jnp.sin(n*T), 0],
                   [jnp.sin(n*T), jnp.cos(n*T), 0],
                   [0, 0, 1]])
    
    # use transpose of R to convert from inertial to rotating frame
    a2 = R.T @ a_SCI

    return t2, a2

def BCP2toBCP1(t2,a2):
    # This function converts a vector from the BCP2 frame to the BCP1 frame

    # transform time t2 to dimensional units
    T = t2 * TU2

    # transform T into nondimensional BCP1 time units
    t1 = T / TU1

    # assume that time T is linearly related to θ_S to get angle of the Sun 
    θ_S = (n3-1)*T # rad

    # angle of the Moon in BCP2
    θ_M = np.pi - θ_S # rad
    sinθ_M, cosθ_M = jnp.sin(θ_M), jnp.cos(θ_M)

    # rotation matrix from BCP1 to BCP2
    C = jnp.array([[cosθ_M, -sinθ_M, 0],
          [sinθ_M, cosθ_M, 0],
          [0, 0, 1]])
    
    # apply transpose of C to go from BCP2 to BCP1
    a1 = C.T @ a2

    return t1, a1