"""
helpers.py

Utility functions for the Bayesian inverse heat conduction challenge.

This module provides:
- Data loading
- Finite difference solver for the Laplace equation
- Mollified point evaluation
- Forward model construction

All functions are written to be reusable and notebook-agnostic.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Plot configuration (safe to import globally)
# ---------------------------------------------------------------------
plt.rcParams.update({"figure.figsize": (5, 4), "font.size": 11})


# ---------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------
def load_data(path: str):
    """
    Load measurement and reference data from .npz file.

    Parameters
    ----------
    path : str
        Path to measurements.npz

    Returns
    -------
    x_meas, y_meas, u_meas : np.ndarray
        Interior measurement coordinates and temperatures.
    sigma : float
        Estimated noise level.
    x_grid, y_grid : np.ndarray
        Grid points used for plotting / FD discretization.
    f_true_y, f_true_vals : np.ndarray
        Sparse reference samples of the true boundary function (comparison only).
    """
    data = np.load(path, allow_pickle=True)

    x_meas = data["x_meas"]
    y_meas = data["y_meas"]
    u_meas = data["u_meas"]
    sigma = float(data["sigma"])

    x_grid = data["x_plot"]
    y_grid = data["y_plot"]

    f_true_y = data["f_true_y"]
    f_true_vals = data["f_true_vals"]

    return x_meas, y_meas, u_meas, sigma, x_grid, y_grid, f_true_y, f_true_vals


def overview_plots(x_meas, y_meas, u_meas, f_true_y, f_true_vals):
    """
    Generate overview plots:
    1) Interior measurement locations colored by temperature
    2) Sparse reference samples of the true boundary f(y)

    Note: boundary samples are for comparison only.
    """
    # Interior measurements
    plt.figure()
    sc = plt.scatter(x_meas, y_meas, c=u_meas, s=40, cmap="viridis")
    plt.colorbar(sc, label="u_meas")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("Interior measurement locations")
    plt.tight_layout()
    plt.show()

    # Boundary reference samples
    plt.figure()
    plt.grid(True, alpha=0.3)
    plt.scatter(f_true_y, f_true_vals)
    plt.xlabel("y (left boundary x = 0)")
    plt.ylabel("Temperature")
    plt.title("Sparse reference samples of f(y)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# PDE solver
# ---------------------------------------------------------------------
def solve_laplace_rbf(N, alpha, y_bc, rbf_width):
    """
    Solves the Laplace equation with a spatially varying Dirichlet BC on the left boundary x = 0:

        u(0, y) = sum_k alpha_k * RBF(y; centre_k, width)
        u(1, y) = 0
        u(x, 0) = 0
        u(x, 1) = 0

    Parameters:
    N (int): Number of interior grid points along one axis.
    alpha (array-like): Coefficients for RBF expansion.
    y_bc (array-like): Locations of the RBF centers along y 
    rbf_width (float): width of the RBF
        Width (standard deviation) of Gaussian RBFs.

    Returns:
    u_grid : 2D numpy array, shape (N+2, N+2). Solution grid including boundaries.
    """
    alpha = np.array(alpha, dtype=float)
    n_points = N * N

    def idx(i, j):
        """Convert (i, j) indices to a flat index."""
        return i * N + j

    # evaluate left boundary (x = 0) (via provided RBF basis expansion)
    y_vals = np.linspace(0, 1, N + 2)  
    left_bc = np.zeros_like(y_vals)
    for k, y in enumerate(y_bc):
        left_bc += alpha[k] * np.exp(-0.5 * ((y_vals - y) / rbf_width) ** 2)

    # Create sparse matrix for Laplacian
    A = sp.lil_matrix((n_points, n_points))
    b = np.zeros(n_points)

    for i in range(N):      # interior x indices
        for j in range(N):  # interior y indices
            k = idx(i, j)
            A[k, k] = -4.0

            # Left neighbour (x = 0): inhomogeneous boundary u = left_bc
            if i > 0:
                A[k, idx(i - 1, j)] = 1.0
            else:
                # interior y index j corresponds to y-index j+1 in left_bc
                b[k] -= left_bc[j + 1]

            # Right neighbour (x = 1): homogeneous boundary u = 0
            if i < N - 1:
                A[k, idx(i + 1, j)] = 1.0
            else:
                b[k] -= 0.0

            # BOTTOM neighbor (y = 0) -> homogeneous boundary u = 0
            if j > 0:
                A[k, idx(i, j - 1)] = 1.0
            else:
                b[k] -= 0.0

            # TOP neighbor (y = 1) -> homogeneous boundary u = 0
            if j < N - 1:
                A[k, idx(i, j + 1)] = 1.0
            else:
                b[k] -= 0.0

    A_csr = A.tocsr()
    u_int = spla.spsolve(A_csr, b)

    # Full grid including boundaries; grid[i,j] corresponds to (x_i, y_j)
    u_grid = np.zeros((N + 2, N + 2))
    u_grid[1:-1, 1:-1] = u_int.reshape((N, N))

    # Apply LEFT boundary at x=0  → index 0 along axis 0
    u_grid[0, :] = left_bc

    # Right, top, bottom boundaries remain zero
    return u_grid



# ---------------------------------------------------------------------
# Observation operator
# ---------------------------------------------------------------------
def evaluate_solution_mollified(grid, points, epsilon):
    """
    Mollified evaluation of a grid-based solution at arbitrary points
    using a Gaussian kernel.

    Parameters
    ----------
    grid : np.ndarray
        Solution grid including boundaries.
    points : list of tuple
        (x, y) evaluation points in [0,1]^2.
    epsilon : float
        Mollifier standard deviation.

    Returns
    -------
    values : np.ndarray
        Mollified solution values at given points.
    """
    N_with_boundary = grid.shape[0]
    x_vals = np.linspace(0, 1, N_with_boundary)
    y_vals = np.linspace(0, 1, N_with_boundary)

    values = []
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    for x0, y0 in points:
        # Compute Gaussian weights centered at (x0, y0)
        dist_squared = (X - x0)**2 + (Y - y0)**2
        weights = np.exp(-dist_squared / (2 * epsilon**2))

        # Normalise weights
        weights /= np.sum(weights)

        # Weighted average of the grid values
        u_mollified = np.sum(weights * grid)
        values.append(u_mollified)

    return values


def forward_model(alpha, measurement_points, N, centres, ell, epsilon=0.03):
    """
    Deterministic forward map alpha -> u_tau(alpha).

    Parameters
    ----------
    alpha : np.ndarray
        Boundary coefficients.
    measurement_points : list of tuple
        Interior measurement locations.
    N : int
        Number of interior FD points.
    centres : np.ndarray
        RBF centers.
    ell : float
        RBF width.
    epsilon : float
        Mollifier width.

    Returns
    -------
    np.ndarray
        Model predictions at measurement points.
    """
    grid = solve_laplace_rbf(N, alpha, centres, ell)
    return evaluate_solution_mollified(grid, measurement_points, epsilon)

def forward_blackbox(alpha, epsilon=0.03):
    """
    Black-box forward map: alpha -> u(alpha) evaluated at measurement_points.

    This assumes that the following globals are defined in the notebook:
    - N_blackbox      : number of interior grid points (N in solve_laplace_rbf)
    - centres, ell    : RBF parameters
    - measurement_points : list of (x,y) interior measurement locations

    Returns
    -------
    np.ndarray: Array of (mollified) model responses u_{tau}[alpha] evaluated at the interior measurement points.
    """
    grid = solve_laplace_rbf(N_blackbox, alpha, centres, ell)
    values = evaluate_solution_mollified(grid, measurement_points, epsilon)
    return np.asarray(values, dtype=float)