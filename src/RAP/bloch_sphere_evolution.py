# File: bloch_sphere_evolution.py
# Created: 01-02-2026
# Author: Andrea Turci <turci.andrea01@gmail.com>
# Institution: University of Innsbruck - UIBK

"""
Module for Simulating and Visualizing RAP Dynamics on the Bloch Sphere.

This module provides tools to:
- Define the time-dependent Rabi rate (Omega) and detuning (Delta) for a RAP pulse.
- Solve the Optical Bloch Equations using numerical integration.
- Generate a high-quality 3D visualization of the Bloch vector trajectory 
  mapped onto a unit sphere.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

from saving import save_figure_in_images


def Omega(
    t: float, 
    Omega0: float, 
    T: float, 
    sigma: float
) -> float:
    """
    Calculates the Gaussian envelope of the Rabi rate at time t.

    Parameters
    ----------
    t : float
        Current time in the simulation.
    Omega0 : float
        Peak Rabi frequency.
    T : float
        Total duration of the pulse (center is at T/2).
    sigma : float
        Standard deviation of the Gaussian pulse.

    Returns
    -------
    float
        The Rabi frequency at time t.
    """
    return Omega0 * np.exp(-(t - T / 2)**2 / (2 * sigma**2))


def Delta(
    t: float, 
    D: float, 
    T: float
) -> float:
    """
    Calculates the time-dependent detuning (linear chirp) at time t.

    Parameters
    ----------
    t : float
        Current time in the simulation.
    D : float
        Total frequency sweep range (chirp).
    T : float
        Total duration of the pulse.

    Returns
    -------
    float
        The instantaneous detuning at time t.
    """
    # Note: Includes a constant offset of 0.022 MHz as defined in the original logic
    return D / T * (t - T / 2) - 2 * np.pi * 0.022


def bloch_equations(
    t: float, 
    S: List[float], 
    D: float, 
    T: float, 
    Omega0: float, 
    sigma: float
) -> List[float]:
    """
    Defines the system of Optical Bloch Equations.

    The evolution is described by dS/dt = W x S, where W is the 
    effective torque vector in the rotating frame.

    Parameters
    ----------
    t : float
        Current time.
    S : list
        Current Bloch vector components [Sx, Sy, Sz].
    D : float
        Chirp sweep range.
    T : float
        Total duration.
    Omega0 : float
        Peak Rabi frequency.
    sigma : float
        Pulse width.

    Returns
    -------
    list
        The derivatives [dSx, dSy, dSz].
    """
    Sx, Sy, Sz = S
    
    # Effective torque vector W components
    Wx = Omega(t, Omega0, T, sigma)
    Wy = 0.0
    Wz = Delta(t, D, T)
    
    # Cross product: dS/dt = W x S
    dSx = Wy * Sz - Wz * Sy
    dSy = Wz * Sx - Wx * Sz
    dSz = Wx * Sy - Wy * Sx
    
    return [dSx, dSy, dSz]


def bloch_sphere_evolution(
    D: float, 
    T: float, 
    Omega0: float, 
    sigma: float, 
    filename: str = "bloch_sphere_RAP.svg"
) -> None:
    """
    Solves the Bloch vector trajectory and plots it on a 3D Bloch sphere.

    Parameters
    ----------
    D : float
        Total chirp sweep range.
    T : float
        Total pulse duration.
    Omega0 : float
        Peak Rabi frequency.
    sigma : float
        Gaussian pulse width.
    filename : str, optional
        Output filename for the saved plot. Default is "bloch_sphere_RAP.svg".
    """
    # Initial condition: system starts in the excited state |e> (represented as +z)
    S0 = [0.0, 0.0, 1.0]

    # Numerical integration using Runge-Kutta 45
    t_span = (0, T)
    t_eval = np.linspace(0, T, 10000)
    sol = solve_ivp(
        bloch_equations, 
        t_span, 
        S0, 
        t_eval=t_eval, 
        method='RK45', 
        args=(D, T, Omega0, sigma)
    )

    # Extract trajectory coordinates
    Sx, Sy, Sz = sol.y

    # ===== 3D BLOCH SPHERE PLOTTING =====
    fig = plt.figure(figsize=(28, 26))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.1, alpha=0.1)

    # Plot the simulated trajectory
    ax.plot(Sx, Sy, Sz, color='b', lw=1.5, label='Bloch vector trajectory')
    
    # Mark initial and potential final states
    ax.scatter([0], [0], [1], color='r', s=50, label='Initial state (+z)')
    ax.scatter([0], [0], [-1], color='r', s=50, label='Target state (-z)')

    # Draw Cartesian axes (x, y, z)
    axis_len = 1.0
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='k', linewidth=2, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='k', linewidth=2, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='k', linewidth=2, arrow_length_ratio=0.08)

    # Axis text labels
    ax.text(axis_len * 2.05, 0, 0, 'x', color='k', fontsize=14)
    ax.text(0, axis_len * 2.05, 0, 'y', color='k', fontsize=14)
    ax.text(0, 0, axis_len * 2.05, 'z', color='k', fontsize=14)

    # Generate the sphere surface for visualization
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        sphere_x, sphere_y, sphere_z,
        color='lightgray',
        alpha=0.15,
        linewidth=0,
        zorder=0
    )

    # Draw the equator line
    phi = np.linspace(0, 2 * np.pi, 400)
    x_eq = np.cos(phi)
    y_eq = np.sin(phi)
    z_eq = np.zeros_like(phi)

    ax.plot(
        x_eq, y_eq, z_eq,
        color='k',
        linestyle='--',
        linewidth=1.5,
        alpha=0.8,
        label='Equator (z = 0)'
    )

    # Axis formatting
    ax.set_xlabel('Sx', fontsize=26)
    ax.set_ylabel('Sy', fontsize=26)
    ax.set_zlabel('Sz', fontsize=26)
    ax.set_title('Bloch sphere and RAP dynamics')
    ax.legend()
    
    # Maintain aspect ratio and set tick parameters
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(axis='both', which='major', labelsize=26, pad=50)

    # Set consistent limits for the unit sphere
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Adjust viewing perspective
    ax.view_init(elev=25, azim=45)

    # Save and display the result
    save_figure_in_images(fig, filename)
    plt.show()