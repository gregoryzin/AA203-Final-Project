"""
Solution code for the problem "Cart-pole balance".

Autonomous Systems Lab (ASL), Stanford University
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from animations import animate_cartpole

# Constants
n = 4  # state dimension
m = 1  # control dimension
mp = 2.0  # pendulum mass, kg 
mc = 10.0  # cart mass, kg
L = 1.0  # pendulum length
g = 9.81  # gravitational acceleration
dt = 0.1  # discretization time step
animate = True  # whether or not to animate results


def cartpole(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Compute the cart-pole state derivative

    Args:
        s (np.ndarray): The cartpole state: [x, theta, x_dot, theta_dot], shape (n,)
        u (np.ndarray): The cartpole control: [F_x], shape (m,)

    Returns:
        np.ndarray: The state derivative, shape (n,)
    """
    x, θ, dx, dθ = s
    sinθ, cosθ = np.sin(θ), np.cos(θ)
    h = mc + mp * (sinθ**2)
    ds = np.array(
        [
            dx,
            dθ,
            (mp * sinθ * (L * (dθ**2) + g * cosθ) + u[0]) / h,
            -((mc + mp) * g * sinθ + mp * L * (dθ**2) * sinθ * cosθ + u[0] * cosθ)
            / (h * L),
        ]
    )
    return ds


def reference(t: float) -> np.ndarray:
    """Compute the reference state (s_bar) at time t

    Args:
        t (float): Evaluation time

    Returns:
        np.ndarray: Reference state, shape (n,)
    """
    a = 10.0  # Amplitude
    T = 10.0  # Period

    # PART (d) ##################################################
    # INSTRUCTIONS: Compute the reference state for a given time
    # raise NotImplementedError()
    
    # intialize reference trajectory
    s_ref = np.zeros((t.size,4))

    # pendulum to be upright
    theta_ref = np.pi
    thetadot_ref = 0

    # want sinusoidal motion in x
    x_ref = a*np.sin(2*np.pi*t/T)
    xdot_ref = a*(2*np.pi/T)*np.cos(2*np.pi*t/T)

    # print(x_ref[0])
    s_ref = [x_ref, theta_ref, xdot_ref, thetadot_ref]

    return s_ref

    # END PART (d) ##############################################


def ricatti_recursion(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Compute the gain matrix K through Ricatti recursion

    Args:
        A (np.ndarray): Dynamics matrix, shape (n, n)
        B (np.ndarray): Controls matrix, shape (n, m)
        Q (np.ndarray): State cost matrix, shape (n, n)
        R (np.ndarray): Control cost matrix, shape (m, m)

    Returns:
        np.ndarray: Gain matrix K, shape (m, n)
    """
    eps = 1e-4  # Riccati recursion convergence tolerance
    max_iters = 1000  # Riccati recursion maximum number of iterations
    P_prev = np.zeros((n, n))  # initialization
    converged = False

    # print(P_prev)

    for i in range(max_iters):
        # PART (b) ##################################################
        # INSTRUCTIONS: Apply the Ricatti equation until convergence
        
        # print(R + B.T @ P_prev @ B)
        # print(np.linalg.inv(R + B.T @ P_prev @ B))
        # print(B.T @ P_prev @ A)
        # print(Q.shape)

        if R.shape == (1,1):
            K = -1 * B.T @ P_prev @ A / (R + B.T @ P_prev @ B)
        else:
            K = -1 @ np.linalg.inv(R + B.T @ P_prev @ B) @ B.T @ P_prev @ A

        # B is shape n,m (4,1)
        # K is shape m,n (1,4)
        # B * K will be shape n,n - need to implement correctly
        # print(K.shape)
        # print(B.reshape(n,m))
        # print(Q + A.T @ P_prev @ (A + B.reshape(n,m) * K))

        P = Q + A.T @ P_prev @ (A + B.reshape(n,m) * K)

        if np.max(np.abs(P - P_prev)) < eps:
            converged = True

        # store P to P_prev
        P_prev = P

        # raise NotImplementedError()
        # END PART (b) ##############################################
    if not converged:
        raise RuntimeError("Ricatti recursion did not converge!")
    print("K:", K)
    return K


def simulate(
    t: np.ndarray, s_ref: np.ndarray, u_ref: np.ndarray, s0: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the cartpole

    Args:
        t (np.ndarray): Evaluation times, shape (num_timesteps,)
        s_ref (np.ndarray): Reference state s_bar, evaluated at each time t. Shape (num_timesteps, n)
        u_ref (np.ndarray): Reference control u_bar, shape (m,)
        s0 (np.ndarray): Initial state, shape (n,)
        K (np.ndarray): Feedback gain matrix (Ricatti recursion result), shape (m, n)

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of:
            np.ndarray: The state history, shape (num_timesteps, n)
            np.ndarray: The control history, shape (num_timesteps, m)
    """

    def cartpole_wrapper(s, t, u):
        """Helper function to get cartpole() into a form preferred by odeint, which expects t as the second arg"""
        return cartpole(s, u)

    # PART (c) ##################################################
    # INSTRUCTIONS: Complete the function to simulate the cartpole system
    # Hint: use the cartpole wrapper above with odeint

    # initialize arrays
    s = np.zeros(s_ref.shape)
    u = np.zeros((t.size,1))

    # print(s[0].shape)
    # print(s0.shape)
    print(s_ref[0])

    # intial state 
    s[0] = s0

    for k in range(t.size - 1):
        # control at index k
        u[k] = K @ (s[k] - s_ref[k]) # + u_ref

        # print(u[k])
        # simulate nonlinear dynamics
        s[k+1] = odeint(cartpole_wrapper, s[k], t[k:k+2], (u[k],))[1]

    # END PART (c) ##############################################
    return s, u


def compute_lti_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Compute the linearized dynamics matrices A and B of the LTI system

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of:
            np.ndarray: The A (dynamics) matrix, shape (n, n)
            np.ndarray: The B (controls) matrix, shape (n, m)
    """
    # PART (a) ##################################################
    # INSTRUCTIONS: Construct the A and B matrices
    # initialize states to linearize around
    # A = eye(4) + jax.jacobian(lambda x: f(x,u))(s) 
    # B = jax.jacobian(lambda v: f(x,v))(u) 
    A = np.eye(4) + dt*np.array([[0,0,1,0],
                                [0,0,0,1],
                                [0,mp*g/mc,0,0],
                                [0,(mp+mc)*g/(mc*L),0,0]])
    
    B = np.array([0, 0, 1/mc, 1/(mc*L)]) * dt

    print(A)
    print(B)

    # END PART (a) ##############################################
    return A, B


def plot_state_and_control_history(
    s: np.ndarray, u: np.ndarray, t: np.ndarray, s_ref: np.ndarray, name: str
) -> None:
    """Helper function for cartpole visualization

    Args:
        s (np.ndarray): State history, shape (num_timesteps, n)
        u (np.ndarray): Control history, shape (num_timesteps, m)
        t (np.ndarray): Times, shape (num_timesteps,)
        s_ref (np.ndarray): Reference state s_bar, evaluated at each time t. Shape (num_timesteps, n)
        name (str): Filename prefix for saving figures
    """
    fig, axes = plt.subplots(n+m, 1, dpi=150, figsize=(2,15))
    plt.subplots_adjust(wspace=0.35)
    labels_s = (r"$x(t)$", r"$\theta(t)$", r"$\dot{x}(t)$", r"$\dot{\theta}(t)$")
    labels_u = (r"$u(t)$",)
    for i in range(n):
        axes[i].plot(t, s[:, i])
        axes[i].plot(t, s_ref[:, i], "--")
        axes[i].set_xlabel(r"$t$")
        axes[i].set_ylabel(labels_s[i])
    for i in range(m):
        axes[n + i].plot(t, u[:, i])
        axes[n + i].set_xlabel(r"$t$")
        axes[n + i].set_ylabel(labels_u[i])
    plt.savefig(f"{name}.png", bbox_inches="tight")
    # plt.show()

    if animate:
        fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
        ani.save(f"{name}.mp4", writer="ffmpeg")
        # plt.show()


def main():
    # Part A
    A, B = compute_lti_matrices()

    # Part B
    Q = np.eye(n)  # state cost matrix
    R = np.eye(m)  # control cost matrix
    K = ricatti_recursion(A, B, Q, R)

    # Part C
    t = np.arange(0.0, 30.0, 1 / 10)
    s_ref = np.array([0.0, np.pi, 0.0, 0.0]) * np.ones((t.size, 1))
    u_ref = np.array([0.0])
    s0 = np.array([0.0, 3 * np.pi / 4, 0.0, 0.0])
    s, u = simulate(t, s_ref, u_ref, s0, K)
    plot_state_and_control_history(s, u, t, s_ref, "cartpole_balance")

    # Part D
    # Note: t, u_ref unchanged from part c
    s_ref = np.array([reference(ti) for ti in t])
    s0 = np.array([0.0, np.pi, 0.0, 0.0])
    # s0 = np.array([0.0, 3 * np.pi / 4, 0.0, 0.0])
    s, u = simulate(t, s_ref, u_ref, s0, K)
    plot_state_and_control_history(s, u, t, s_ref, "cartpole_balance_tv")


if __name__ == "__main__":
    main()
