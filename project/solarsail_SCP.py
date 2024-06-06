"""
Code for AA 203 Final Project using SCP - Gregory Zin

Autonomous Systems Lab (ASL), Stanford University
"""

from functools import partial

from helper import *

import cvxpy as cvx

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm


@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : jax.numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : jax.numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    c : jax.numpy.ndarray
        The offset term in the first-order Taylor expansion of `f` at `(s, u)`
        that sums all vector terms strictly dependent on the nominal point
        `(s, u)` alone.
    """
    # PART (b) ################################################################
    # INSTRUCTIONS: Use JAX to affinize `f` around `(s, u)` in two lines.
    # raise NotImplementedError()
    A,B = jax.jacrev(f,argnums = (0,1))(s,u)
    # A = jax.jacobian(lambda x: f(x,u))(s) 
    # B = jax.jacobian(lambda v: f(s,v))(u) 
    c = f(s,u) - A @ s - B @ u  
    # END PART (b) ############################################################
    return A, B, c


def solve_swingup_scp(f, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters):
    """Solve the cart-pole swing-up problem via SCP.

    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    ρ : float
        Trust region radius.
    eps : float
        Termination threshold for SCP.
    max_iters : int
        Maximum number of SCP iterations.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : numpy.ndarray
        A 1-D array where `J[i]` is the SCP sub-problem cost after the i-th
        iteration, for `i = 0, 1, ..., (iteration when convergence occured)`
    """
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize dynamically feasible nominal trajectories
    u = np.zeros((N, m))
    s = np.zeros((N + 1, n))
    s[0] = s0
    for k in range(N):
        # use heuristic initial guess
        u[k] = np.array([-np.pi/4, 0])

        s[k + 1] = fd(s[k], u[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, N, P, Q, R, u_max, ρ)
        dJ = np.abs(J[i + 1] - J[i])
        prog_bar.set_postfix({"objective change": "{:.5f}".format(dJ)})
        if dJ < eps:
            converged = True
            print("SCP converged after {} iterations.".format(i))
            break
    if not converged:
        raise RuntimeError("SCP did not converge!")
    J = J[1 : i + 1]
    return s, u, J


def scp_iteration(f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, ρ):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem.

    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    s_prev : numpy.ndarray
        The state trajectory around which the problem is convexified (2-D).
    u_prev : numpy.ndarray
        The control trajectory around which the problem is convexified (2-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    ρ : float
        Trust region radius.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : float
        The SCP sub-problem cost.
    """
    A, B, c = affinize(f, s_prev[:-1], u_prev)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    m = R.shape[0]
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # PART (c) ################################################################
    # INSTRUCTIONS: Construct the convex SCP sub-problem.

    objective = (cvx.quad_form(s_cvx[-1] - s_goal,P) +
                 cvx.sum([cvx.quad_form(u_cvx[i],R) for i in range(N)]) +
                cvx.sum([cvx.quad_form(s_cvx[i] - s_goal,Q) for i in range(N)])) 
    
    constraints = [s_cvx[i+1] == c[i] + A[i] @ s_cvx[i] + B[i] @ u_cvx[i] for i in range(N)] # dynamics constraint 
    constraints += [s_cvx[0] == s0] # initial/terminal constraints 
    constraints += [cvx.abs(u_cvx) <= u_max] # control constraint 
    
    # trust region constraints
    constraints += [cvx.norm_inf(s_cvx[i] - s_prev[i]) <= ρ for i in range(N)]  
    constraints += [cvx.norm_inf(u_cvx[i] - u_prev[i]) <= ρ for i in range(N)]

    # raise NotImplementedError()
    # END PART (c) ############################################################

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()
    # prob.solve(solver=cvx.ECOS, verbose=True)
    if prob.status != "optimal":
        raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J


def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 1.0  # pendulum mass
    mc = 4.0  # cart mass
    L = 1.0  # pendulum length
    g = 9.81  # gravitational acceleration

    x, θ, dx, dθ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    h = mc + mp * (sinθ**2)
    ds = jnp.array(
        [
            dx,
            dθ,
            (mp * sinθ * (L * (dθ**2) + g * cosθ) + u[0]) / h,
            -((mc + mp) * g * sinθ + mp * L * (dθ**2) * sinθ * cosθ + u[0] * cosθ)
            / (h * L),
        ]
    )
    return ds


def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator



# Define constants
n = 7  # state dimension
m = 2  # control dimension
P = 1e3 * np.eye(n)  # terminal state cost matrix
Q = np.diag(np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))  # state cost matrix
R = 1e-2 * np.eye(m)  # control cost matrix
s0 = np.array([1.016199666777148,   0.043953600688565,  -0.114729547537933,   0.070089213834930,  -0.029738753389694,  -0.284151852739576,  -7.205475689979612])  # initial state
s_goal = np.array([a3, 0.0, 0.0, 0.0, a3*(n3-1), 0.0, -200*np.pi])  # goal state
T = (s_goal[-1] - s0[-1])/(n3-1)  # simulation time
dt = 0.01  # sampling time

ρ = 1.0  # trust region parameter
u_max = np.pi/2  # control effort bound
eps = 5e-1  # convergence tolerance
max_iters = 100  # maximum number of SCP iterations
# animate = True  # flag for animation

# Initialize the discrete-time dynamics
fd = jax.jit(discretize(BCR4BP_SRP, dt))

# Solve the swing-up problem with SCP
t = np.arange(0.0, T + dt, dt)
N = t.size - 1
s, u, J = solve_swingup_scp(fd, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters)

# Simulate open-loop control
for k in range(N):
    s[k + 1] = fd(s[k], u[k])

# Plot state and control trajectories
fig, axes = plt.subplots(3, 3, dpi=150, figsize=(10, 10))
plt.subplots_adjust(wspace=0.45)
labels_s = (r"$x(t)$", r"$y(t)$", r"$z(t)$", r"$\dot{x}(t)$", r"$\dot{y}(t)$", r"$\dot{z}(t)$", r"$\theta(t)$")
labels_u = (r"$\alpha(t)$",r"$\beta(t)$")
for i in range(n):
    axes[int(np.floor(i/3)),np.mod(i,3)].plot(t, s[:, i])
    axes[int(np.floor(i/3)),np.mod(i,3)].set_xlabel(r"$t$")
    axes[int(np.floor(i/3)),np.mod(i,3)].set_ylabel(labels_s[i])
for i in range(m):
    axes[int(np.floor((n + i)/3)),np.mod(n + i,3)].plot(t[:-1], u[:, i])
    axes[int(np.floor((n + i)/3)),np.mod(n + i,3)].set_xlabel(r"$t$")
    axes[int(np.floor((n + i)/3)),np.mod(n + i,3)].set_ylabel(labels_u[i])
plt.savefig("solarsail_SCP.png", bbox_inches="tight")
plt.show()

# Plot cost history over SCP iterations
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 5))
ax.semilogy(J)
ax.set_xlabel(r"SCP iteration $i$")
ax.set_ylabel(r"SCP cost $J(\bar{x}^{(i)}, \bar{u}^{(i)})$")
plt.savefig("solarsail_constrained_cost.png", bbox_inches="tight")
plt.show()

# Animate the solution
# if animate:
#     fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
#     ani.save("cartpole_swingup_constrained.gif", writer="ffmpeg")
#     plt.show()
