import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from scipy.sparse.linalg import cg
from time import time
import pandas as pd



# Define the domain Omega and discretize
N = 100  # Number of grid points
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
dx, dy = x[1] - x[0], y[1] - y[0]
dx2, dy2 = dx**2, dy**2

# Source function and heterogeneity
g = -6 * X * Y**2 * (X**2 + 2 * Y**2) + 5 * X**3 * Y**4
k_base = 10 # Base wave number
k1, k2, k3 = 2 * k_base, 1.5 * k_base, 3 * k_base
d1, d2, d3, d4 = 0.5, 0.4, 0.6, 0.8
lambda_helmholtz = np.zeros_like(X, dtype=complex)
lambda_helmholtz[Y <= d1], lambda_helmholtz[(Y > d1) & (Y <= d2)] = k1**2, k2**2
lambda_helmholtz[(Y > d2) & (Y <= d3)], lambda_helmholtz[Y > d3] = k3**2, k_base**2

# Construct the Laplacian operator
laplacian_x = diags([-2*np.ones(N), np.ones(N-1), np.ones(N-1)], [0, -1, 1], shape=(N, N)) / dx2
laplacian_y = diags([-2*np.ones(N), np.ones(N-1), np.ones(N-1)], [0, -1, 1], shape=(N, N)) / dy2
laplacian_2d = kron(identity(N), laplacian_x) + kron(laplacian_y, identity(N))

# Helmholtz operator and boundary conditions
helmholtz_operator = laplacian_2d - diags(lambda_helmholtz.flatten(), 0)
u_top, u_right = x**3, y**4
u_bottom, u_left = np.zeros(N), np.zeros(N)
g_1d = g.flatten()
g_1d[-N:] -= u_top / dy2
g_1d[::N] -= u_left / dx2
g_1d[N-1::N] -= u_right / dx2
g_1d[:N] -= u_bottom / dy2

# Solve the Helmholtz equation
u_1d = spsolve(helmholtz_operator, g_1d)
u = u_1d.reshape((N, N))

# Analytical solution and error
u_analytical = X**3 * Y**4
error = u - u_analytical

# Calculate the residual
u_xx = np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)
u_yy = np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)
u_xx /= dx2
u_yy /= dy2
residual = - (u_xx + u_yy) - lambda_helmholtz * u - g



# Define a callback function to store iteration data
iter_data = {'Iteration': [], 'CPU Time': [], 'Error': []}

def callback(xk):
    iter_num = len(iter_data['Iteration']) + 1
    iter_data['Iteration'].append(iter_num)
    iter_data['CPU Time'].append(time() - start_time)
    u_current = xk.reshape((N, N))
    error_current = np.linalg.norm(u_current - u_analytical)
    iter_data['Error'].append(error_current)

# Start timing
start_time = time()

# Solve the system using an iterative solver
u_1d, _ = cg(helmholtz_operator, g_1d, callback=callback)

# Stop timing
end_time = time()

# Reshape the final solution
u = u_1d.reshape((N, N))

# Create a DataFrame for the iteration data
df = pd.DataFrame(iter_data)

# Print the DataFrame
print(df)


# Interactive Surface Plot for the Solution
fig_solution = go.Figure(data=[go.Surface(z=u.real, x=X, y=Y)])
fig_solution.update_layout(title='Numerical Solution', autosize=False,
                           width=600, height=600,
                           margin=dict(l=65, r=50, b=65, t=90))
fig_solution.show()

# Interactive Surface Plot for the Residual
fig_residual = go.Figure(data=[go.Surface(z=residual.real, x=X, y=Y)])
fig_residual.update_layout(title='Residual Error', autosize=False,
                           width=600, height=600,
                           margin=dict(l=65, r=50, b=65, t=90))
fig_residual.show()


# Compute Error Metrics
mse = np.mean((u.real - u_analytical)**2)
max_abs_error = np.max(np.abs(u.real - u_analytical))

print("Mean Squared Error:", mse)
print("Maximum Absolute Error:", max_abs_error)


# Create a figure and a 3D subplot
#fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
# Adjusting the size of the subplot for numerical and analytical solutions
fig, ax = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={'projection': '3d'})  # Increased size


# Plotting for comparison
#fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 6))

# Numerical Solution
ax[0].plot_surface(X, Y, u.real, cmap='viridis')
ax[0].set_title('Numerical Solution')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_zlabel('u')

# Analytical Solution
ax[1].plot_surface(X, Y, u_analytical, cmap='viridis')
ax[1].set_title('Analytical Solution')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_zlabel('u')

plt.tight_layout()

# Plotting
fig = plt.figure(figsize=(14, 4))
'''
# Numerical Solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Y, u.real, cmap='viridis', edgecolor='none')
ax1.set_title('Numerical Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
'''

# Residual Error
ax2 = fig.add_subplot(131, projection='3d')
surf2 = ax2.plot_surface(X, Y, residual.real, cmap='viridis', edgecolor='none')
ax2.set_title('Residual Error')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Residual')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# Error (Numerical - Analytical)
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X, Y, error.real, cmap='viridis', edgecolor='none')
ax3.set_title('Error (Numerical - Analytical)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Error')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

plt.show()
