"""
    Created by:
        Opal Issan

    Modified:
        17 Nov 2020 - Jay Lago

    Modified:
        31 Dec 2022 - Francesco Andreuzzi

    Source: https://github.com/JayLago/DLDMD
"""
import numpy as np


# ==============================================================================
# Function Implementations
# ==============================================================================
def dyn_sys_discrete(lhs, mu=-0.05, lam=-1):
    """ example 1:
    ODE =>
    dx1/dt = mu*x1
    dx2/dt = lam*(x2-x1^2)

    By default: mu =-0.05, and lambda = -1.
    """
    rhs = np.zeros(2)
    rhs[0] = mu * lhs[0]
    rhs[1] = lam * (lhs[1] - (lhs[0]) ** 2.)
    return rhs

def dyn_sys_pendulum(lhs):
    """ pendulum example:
    ODE =>
    dx1/dt = x2
    dx2/dt = -sin(x1)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -np.sin(lhs[0])
    return rhs

def dyn_sys_fluid(lhs, mu=0.1, omega=1, A=-0.1, lam=10):
    """fluid flow example:
    ODE =>
    dx1/dt = mu*x1 - omega*x2 + A*x1*x3
    dx2/dt = omega*x1 + mu*x2 + A*x2*x3
    dx3/dt = -lam(x3 - x1^2 - x2^2)
    """
    rhs = np.zeros(3)
    rhs[0] = mu * lhs[0] - omega * lhs[1] + A * lhs[0] * lhs[2]
    rhs[1] = omega * lhs[0] + mu * lhs[1] + A * lhs[1] * lhs[2]
    rhs[2] = -lam * (lhs[2] - lhs[0] ** 2 - lhs[1] ** 2)
    return rhs

def dyn_sys_kdv(lhs, a1=0, c=3):
    """ planar kdv:
    dx1/dt = x2
    dx2/dt = a1 + c*x1 - 3*x2^2
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = a1 + c*lhs[0] - 3*lhs[0]**2
    return rhs

def dyn_sys_duffing_driven(lhs, alpha=0.1, gamma=0.05, omega=1.1):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = x - x^3 - gamma*y + alpha*cos(omega*t)
    """
    rhs = np.zeros(3)
    rhs[0] = lhs[1]
    rhs[1] = lhs[0] - lhs[0]**3 - gamma*lhs[1] + alpha*np.cos(omega*lhs[2])
    rhs[2] = lhs[2]
    return rhs

def dyn_sys_duffing(lhs):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = x - x^3
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = lhs[0] - lhs[0]**3
    return rhs

def dyn_sys_duffing_bollt(lhs, alpha=1.0, beta=-1.0, delta=0.5):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = -delta*y - x*(beta + alpha*x^2)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -delta*lhs[1] - lhs[0]*(beta + alpha*lhs[0]**2)
    return rhs

def rk4(lhs, dt, function):
    """
    :param lhs: previous step state.
    :param dt: delta t.
    :param data_type: "ex1" or "ex2".
    :return:  Runge–Kutta 4th order method.
    """
    k1 = dt * function(lhs)
    k2 = dt * function(lhs + k1 / 2.0)
    k3 = dt * function(lhs + k2 / 2.0)
    k4 = dt * function(lhs + k3)
    rhs = lhs + 1.0 / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return rhs

def data_maker_discrete(x_lower1, x_upper1, x_lower2, x_upper2, n_ic=1e4, dt=0.02, tf=1.0, seed=None, testing=False):
    """
    :param tf: final time. default is 15.
    :param dt: delta t.
    :param x_lower1: lower bound of x1, initial condition.
    :param x_upper1: upper bound of x1, initial condition.
    :param x_upper2: lower bound of x2, initial condition.
    :param x_lower2: upper bound of x1, initial condition.
    :param n_side: number of initial conditions on each axis. default is 100.
    :return:
    """
    # set seed
    np.random.seed(seed=seed)

    # dim - time steps
    nsteps = int(tf / dt)

    # number of initial conditions.
    n_ic = int(n_ic)

    # create initial condition grid
    if testing:
        icond1 = np.linspace(x_lower1, x_upper1, 10)
        icond2 = np.linspace(x_lower2, x_upper2, 2)
        xx, yy = np.meshgrid(icond1, icond2)

        # solve the system using Runge–Kutta 4th order method, see rk4 function above.
        data_mat = np.zeros((n_ic, 2, nsteps + 1), dtype=np.float64)
        ic = 0
        for x1 in range(2):
            for x2 in range(10):
                data_mat[ic, :, 0] = np.array([xx[x1, x2], yy[x1, x2]], dtype=np.float64)
                for jj in range(nsteps):
                    data_mat[ic, :, jj + 1] = rk4(data_mat[ic, :, jj], dt, dyn_sys_discrete)
                ic += 1
    else:
        icond1 = np.random.uniform(x_lower1, x_upper1, n_ic)
        icond2 = np.random.uniform(x_lower2, x_upper2, n_ic)

        # solve the system using Runge–Kutta 4th order method, see rk4 function above.
        data_mat = np.zeros((n_ic, 2, nsteps + 1), dtype=np.float64)
        for ii in range(n_ic):
            data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
            for jj in range(nsteps):
                data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_discrete)

    return np.transpose(data_mat, [0, 2, 1])

def data_maker_pendulum(x_lower1, x_upper1, x_lower2, x_upper2, n_ic=10000, dt=0.02, tf=1.0, seed=None):
    """
    :param tf: final time. default is 15.
    :param dt: delta t.
    :param x_lower1: lower bound of x1, initial condition.
    :param x_upper1: upper bound of x1, initial condition.
    :param x_upper2: lower bound of x2, initial condition.
    :param x_lower2: upper bound of x1, initial condition.
    :param n_ic: number of initial conditions
    :return:
    """
    # set seed
    np.random.seed(seed=seed)

    # dim - time steps
    nsteps = int(tf / dt)

    # number of initial conditions
    n_ic = int(n_ic)

    # create initial condition grid
    rand_x1 = np.random.uniform(x_lower1, x_upper1, 100 * n_ic)
    rand_x2 = np.random.uniform(x_lower2, x_upper2, 100 * n_ic)
    max_potential = 0.99
    potential = lambda x, y: (1 / 2) * y ** 2 - np.cos(x)
    iconds = np.asarray([[x, y] for x, y in zip(rand_x1, rand_x2)
                         if potential(x, y) <= max_potential])[:n_ic, :]

    # solve the system using Runge–Kutta 4th order method, see rk4 function above
    data_mat = np.zeros((n_ic, 2, nsteps + 1), dtype=np.float64)
    for ii in range(n_ic):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1]], dtype=np.float64)
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_pendulum)

    return np.transpose(data_mat, [0, 2, 1])

def data_maker_pendulum_uniform(n_ic=10000, dt=0.02, tf=3.0, seed=None):
    nsteps = int(tf / dt)
    n_ic = int(n_ic)
    rand_x1 = np.random.uniform(-3.1, 0, 100*n_ic)
    rand_x2 = np.zeros((100*n_ic))
    max_potential = 0.99
    potential = lambda x, y: (1 / 2) * y ** 2 - np.cos(x)
    iconds = np.asarray([[x, y] for x, y in zip(rand_x1, rand_x2)
                         if potential(x, y) <= max_potential])[:n_ic, :]
    data_mat = np.zeros((n_ic, 2, nsteps + 1), dtype=np.float64)
    for ii in range(n_ic):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1]], dtype=np.float64)
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_pendulum)

    return np.transpose(data_mat, [0, 2, 1])

def data_maker_fluid_flow_slow(r_lower=0, r_upper=1.1, t_lower=0, t_upper=2*np.pi, n_ic=1e4, dt=0.05, tf=6, seed=None):
    """
    :param r_lower: lower bound for r. Default is 0.
    :param r_upper: Upper bound for r. Default is 1.
    :param t_lower: Lower bound for theta. Default is 0.
    :param t_upper: Upper bound for theta. Default is 2pi.
    :param n_ic: number of initial conditions. Default is 10000.
    :param dt: time step size. Default is 0.05.
    :param tf: final time. default is 6.
    :return: csv file
    """
    # set seed
    np.random.seed(seed=seed)

    # dim - time steps
    nsteps = int(tf / dt)

    # number of initial conditions for slow manifold.
    n_ic_slow = int(n_ic)

    # create initial condition grid.
    r = np.random.uniform(r_lower, r_upper, n_ic_slow)
    theta = np.random.uniform(t_lower, t_upper, n_ic_slow)

    # compute x1, x2, and x3, based on theta and r
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    x3 = np.power(x1, 2) + np.power(x2, 2)

    # initialize initial conditions matrix.
    iconds = np.zeros((n_ic_slow, 3))

    # initial conditions for slow manifold.
    iconds[:n_ic_slow] = np.asarray([[x, y, z] for x, y, z in zip(x1, x2, x3)])

    # solve the system using Runge–Kutta 4th order method, see rk4 function above.
    data_mat = np.zeros((n_ic_slow, 3, nsteps + 1), dtype=np.float64)
    for ii in range(n_ic_slow):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1], iconds[ii, 2]], dtype=np.float64)
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_fluid)

    return np.transpose(data_mat, [0, 2, 1])

def data_maker_fluid_flow_full(x1_lower=-1.1, x1_upper=1.1, x2_lower=-1.1, x2_upper=1.1, x3_lower=0.0, x3_upper=2.43,
                               n_ic=1e4, dt=0.05, tf=6, seed=None):
    # set seed
    np.random.seed(seed=seed)

    # Number of time steps
    nsteps = int(tf / dt)

    # Number of initial conditions
    n_ic = int(n_ic)

    # Create initial condition grid
    x1 = np.random.uniform(x1_lower, x1_upper, n_ic)
    x2 = np.random.uniform(x2_lower, x2_upper, n_ic)
    x3 = np.random.uniform(x3_lower, x3_upper, n_ic)

    # Initialize initial conditions matrix
    iconds = np.zeros((n_ic, 3))

    # Initial conditions zip
    iconds[:n_ic] = np.asarray([[x, y, z] for x, y, z in zip(x1, x2, x3)])

    # Solve the system using Runge–Kutta 4th order method, see rk4 function above
    data_mat = np.zeros((n_ic, 3, nsteps+1), dtype=np.float64)
    for ii in range(n_ic):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1], iconds[ii, 2]], dtype=np.float64)
        for jj in range(nsteps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_fluid)

    return np.transpose(data_mat, [0, 2, 1])

def data_maker_kdv(x_lower1, x_upper1, x_lower2, x_upper2, n_ic=10000, dt=0.01, tf=1.0, seed=None):
    # Setup
    np.random.seed(seed=seed)
    nsteps = int(tf / dt)
    n_ic = int(n_ic)
    # Generate initial conditions
    icond1 = np.random.uniform(x_lower1, x_upper1, 10*n_ic)
    icond2 = np.random.uniform(x_lower2, x_upper2, 10*n_ic)
    n_try = 10*n_ic
    # Integrate
    data_mat = np.zeros((n_try, 2, nsteps+1), dtype=np.float64)
    for ii in range(n_try):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(nsteps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_kdv)
            # if (data_mat[ii, 0, jj+1] < x_lower1 or data_mat[ii, 1, jj+1] > x_upper1
            #         or data_mat[ii, 1, jj+1] < x_lower2 or data_mat[ii, 1, jj+1] > x_upper2):
            #     break
    accept = np.abs(data_mat[:, 0, -1]) < 3
    data_mat = data_mat[accept, :, :]
    accept = np.abs(data_mat[:, 1, -1]) < 3
    data_mat = data_mat[accept, :, :]
    data_mat = data_mat[:n_ic, :, :]
    return np.transpose(data_mat, [0, 2, 1])

def data_maker_duffing_driven(x_lower1, x_upper1, x_lower2, x_upper2, n_ic=10000, dt=0.01, tf=1.0, seed=None):
    # Setup
    np.random.seed(seed=seed)
    nsteps = int(tf / dt)
    n_ic = int(n_ic)
    # Generate initial conditions
    icond1 = np.random.uniform(x_lower1, x_upper1, n_ic)
    icond2 = np.random.uniform(x_lower2, x_upper2, n_ic)
    # Integrate
    data_mat = np.zeros((n_ic, 3, nsteps+1), dtype=np.float64)
    for ii in range(n_ic):
        data_mat[ii, :2, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(nsteps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_duffing_driven)
            data_mat[ii, 2, jj+1] = data_mat[ii, 2, jj] + dt
    return np.transpose(data_mat, [0, 2, 1])

def data_maker_duffing(x_lower1, x_upper1, x_lower2, x_upper2, n_ic=10000, dt=0.01, tf=1.0, seed=None):
    # Setup
    np.random.seed(seed=seed)
    nsteps = int(tf / dt)
    n_ic = int(n_ic)
    # Generate initial conditions
    icond1 = np.random.uniform(x_lower1, x_upper1, n_ic)
    icond2 = np.random.uniform(x_lower2, x_upper2, n_ic)
    # Integrate
    data_mat = np.zeros((n_ic, 2, nsteps+1), dtype=np.float64)
    for ii in range(n_ic):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(nsteps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_duffing)
    return np.transpose(data_mat, [0, 2, 1])

def data_maker_duffing_bollt(x_lower1, x_upper1, x_lower2, x_upper2, n_ic=10000, dt=0.01, tf=1.0, seed=None):
    # Setup
    np.random.seed(seed=seed)
    nsteps = int(tf / dt)
    n_ic = int(n_ic)
    # Generate initial conditions
    icond1 = np.random.uniform(x_lower1, x_upper1, n_ic)
    icond2 = np.random.uniform(x_lower2, x_upper2, n_ic)
    # Integrate
    data_mat = np.zeros((n_ic, 2, nsteps+1), dtype=np.float64)
    for ii in range(n_ic):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(nsteps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_duffing_bollt)
    return np.transpose(data_mat, [0, 2, 1])

# ==============================================================================
# Test program
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    create_discrete = False
    create_pendulum = False
    create_fluid_flow_slow = False
    create_fluid_flow_full = False
    create_kdv = False
    create_duffing = True

    if create_discrete:
        # Generate the data
        data = data_maker_discrete(x_lower1=-0.5, x_upper1=0.5, x_lower2=-0.5, x_upper2=0.5, n_ic=20, dt=0.02, tf=10)
        # Visualize
        plt.figure(1, figsize=(8, 8))
        for ii in range(data.shape[0]):
            plt.plot(data[ii, :, 0], data[ii, :, 1], '-')
        plt.xlabel("x1", fontsize=18)
        plt.ylabel("X2", fontsize=18)
        plt.title("Discrete dataset", fontsize=18)

    if create_pendulum:
        # Generate the data
        data = data_maker_pendulum(x_lower1=-3.1, x_upper1=3.1, x_lower2=-2, x_upper2=2, n_ic=20, dt=0.02, tf=20)
        # Visualize
        plt.figure(2, figsize=(8, 8))
        for ii in range(data.shape[0]):
            plt.plot(data[ii, :, 0], data[ii, :, 1], '-')
        plt.xlabel("x1", fontsize=18)
        plt.ylabel("X2", fontsize=18)
        plt.title("Pendulum dataset", fontsize=18)

    if create_fluid_flow_slow:
        # Generate the data
        data = data_maker_fluid_flow_slow(r_lower=0, r_upper=1.1, t_lower=0, t_upper=2*np.pi, n_ic=20, dt=0.05, tf=10)
        # Visualize
        fig = plt.figure(3, figsize=(8, 8))
        ax = plt.axes(projection='3d')
        for ii in range(data.shape[0]):
            ax.plot3D(data[ii, :, 0], data[ii, :, 1], data[ii, :, 2])
        ax.set_xlabel("$x_{1}$", fontsize=18)
        ax.set_ylabel("$x_{2}$", fontsize=18)
        ax.set_zlabel("$x_{3}$", fontsize=18)
        plt.title("Fluid Flow dataset", fontsize=20)

    if create_fluid_flow_full:
        # Generate the data
        data = data_maker_fluid_flow_full(x1_lower=-1.1, x1_upper=1.1, x2_lower=-1.1, x2_upper=1.1,
                                          x3_lower=0.0, x3_upper=2.43, n_ic=20, dt=0.05, tf=6)
        # Visualize
        fig = plt.figure(4, figsize=(8, 8))
        ax = plt.axes(projection='3d')
        for ii in range(data.shape[0]):
            ax.plot3D(data[ii, :, 0], data[ii, :, 1], data[ii, :, 2])
        ax.set_xlabel("$x_{1}$", fontsize=18)
        ax.set_ylabel("$x_{2}$", fontsize=18)
        ax.set_zlabel("$x_{3}$", fontsize=18)
        plt.title("Fluid Flow dataset", fontsize=20)

    if create_kdv:
        # Generate the data
        data = data_maker_kdv(x_lower1=-2, x_upper1=2, x_lower2=-2, x_upper2=2, n_ic=1000, dt=0.01, tf=20)
        # Visualize
        plt.figure(2, figsize=(8, 8))
        for ii in range(data.shape[0]):
            npts = np.sum(np.abs(data[ii, :, 0]) > 0)
            plt.plot(data[ii, :npts, 0], data[ii, :npts, 1], 'r-', lw=0.25)
        plt.xlabel("x1", fontsize=18)
        plt.ylabel("X2", fontsize=18)
        plt.title("KdV dataset", fontsize=18)

    if create_duffing:
        # Generate the data
        data = data_maker_duffing(x_lower1=-1, x_upper1=1, x_lower2=-1, x_upper2=1, n_ic=2, dt=0.05, tf=200)
        # Visualize
        plt.figure(2, figsize=(8, 8))
        plt.plot(data[0, :, 0], data[0, :, 1], 'r-', lw=0.5)
        plt.plot(data[1, :, 0], data[1, :, 1], 'b-', lw=0.5)
        plt.xlabel("x1", fontsize=18)
        plt.ylabel("X2", fontsize=18)
        plt.title("Duffing oscillator", fontsize=18)

    plt.show()
    print("done")