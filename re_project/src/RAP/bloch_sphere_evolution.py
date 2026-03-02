import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from saving import save_figure_in_images



# Definizione delle funzioni del RAP pulse
def Omega(t, Omega0, T, sigma):
    return Omega0 * np.exp(-(t - T/2)**2 / (2 * sigma**2))

def Delta(t, D, T):
    return D / T * (t - T/2) - 2 * np.pi * 0.022

# Optical Bloch equations
def bloch_equations(t, S, D, T, Omega0, sigma):
    Sx, Sy, Sz = S
    # effective torque vector W
    Wx = Omega(t, Omega0, T, sigma)
    Wy = 0.0
    Wz = Delta(t, D, T)
    # dS/dt = W x S
    dSx = Wy*Sz - Wz*Sy
    dSy = Wz*Sx - Wx*Sz
    dSz = Wx*Sy - Wy*Sx
    return [dSx, dSy, dSz]


def bloch_sphere_evolution(D, T, Omega0, sigma, filename="bloch_sphere_RAP.svg"):

    # Condizioni iniziali: stato iniziale +z
    S0 = [0.0, 0.0, 1.0]

    # Risoluzione numerica
    t_span = (0, T)
    t_eval = np.linspace(0, T, 10000)
    sol = solve_ivp(bloch_equations, t_span, S0, t_eval=t_eval, method='RK45', args=(D, T, Omega0, sigma))

    # Estrazione risultati
    Sx, Sy, Sz = sol.y

    # ===== PLOT 3D CON SFERA DI BLOCH =====
    fig = plt.figure(figsize=(28, 26))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.1, alpha=0.1)

    # Traiettoria del vettore di Bloch
    ax.plot(Sx, Sy, Sz, color='b', lw=1.5, label='Bloch vector trajectory')
    ax.scatter([0], [0], [1], color='r', s=50, label='Initial state (+z)')
    ax.scatter([0], [0], [-1], color='r', s=50, label='Initial state (-z)')

    # Assi cartesiani (unitari)
    axis_len = 1.0

    ax.quiver(0, 0, 0, axis_len, 0, 0, color='k', linewidth=2, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='k', linewidth=2, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='k', linewidth=2, arrow_length_ratio=0.08)

    # Etichette degli assi
    ax.text(axis_len*2.05, 0, 0, 'x', color='k', fontsize=14)
    ax.text(0, axis_len*2.05, 0, 'y', color='k', fontsize=14)
    ax.text(0, 0, axis_len*2.05, 'z', color='k', fontsize=14)

    # Sfera di Bloch
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        x, y, z,
        color='lightgray',
        alpha=0.15,
        linewidth=0,
        zorder=0
    )

    phi = np.linspace(0, 2*np.pi, 400)
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


    # Assi e formattazione
    ax.set_xlabel('Sx', fontsize = 26)
    ax.set_ylabel('Sy', fontsize = 26)
    ax.set_zlabel('Sz', fontsize = 26)
    ax.set_title('Bloch sphere and RAP dynamics')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(axis='both', which='major', labelsize=26, pad=50)  # dimensione numeri assi
    # for t in ax.zaxis.get_major_ticks():
    #     t.label.set_fontsize(16)

    # Limiti coerenti con la sfera unitaria
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.view_init(elev=25, azim=45)

    save_figure_in_images(fig, filename)

    plt.show()
