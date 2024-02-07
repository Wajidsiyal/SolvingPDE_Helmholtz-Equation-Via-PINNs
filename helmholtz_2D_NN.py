import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants
K_values = [2]  # Wave number scaling factor for the loop
epsilon1 = 0.5  # x-coordinate of Dirac delta source
epsilon2 = 0.8  # y-coordinate of Dirac delta source

# Training parameters
epochs = 1
learning_rate = 0.001


# Define the neural network model for PINNs
def create_PINN_model_2D(layers, activation_function):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,)))  # Input layer for 2D problem
    for layer in layers:
        model.add(tf.keras.layers.Dense(layer, activation=activation_function))
    model.add(tf.keras.layers.Dense(1, activation=None))  # Output layer
    return model


# Boundary Conditions
def apply_2D_boundary_conditions(model, x, y):
    # Create a grid of (x, y) points
    X, Y = tf.meshgrid(x, y)
    X = tf.reshape(X, [-1, 1])
    Y = tf.reshape(Y, [-1, 1])

    # Apply boundary conditions
    u_00_y = model(tf.concat([tf.zeros_like(Y), Y], axis=1))  # u(0, y)
    u_x_0 = model(tf.concat([X, tf.zeros_like(X)], axis=1))  # u(x, 0)
    u_1_y = model(tf.concat([tf.ones_like(Y), Y], axis=1))  # u(1, y)
    u_x_1 = model(tf.concat([X, tf.ones_like(X)], axis=1))  # u(x, 1)

    # Calculate boundary condition losses
    bc_loss_00_y = tf.reduce_mean(tf.square(u_00_y))
    bc_loss_x_0 = tf.reduce_mean(tf.square(u_x_0))
    bc_loss_1_y = tf.reduce_mean(tf.square(u_1_y - tf.pow(Y, 4)))
    bc_loss_x_1 = tf.reduce_mean(tf.square(u_x_1 - tf.pow(X, 3)))

    return bc_loss_00_y + bc_loss_x_0 + bc_loss_1_y + bc_loss_x_1


# PINNS loss for 2D heterogeneity in wave number k

# Helmholtz 2D model with g(x,y)=−6xy2 (x2 + 2y2 ) + 5x3 y4


    

def PINNs_loss_2D(model, x, y,k):
    # Define the heterogeneity in wave number k
    def wave_number(x, y):
        k1 = 2.0  # First region wave number scaling factor
        k2 = 1.5  # Second region wave number scaling factor
        k3 = 3.0  # Third region wave number scaling factor

        # Calculate the wave number based on the domain regions
        k = tf.where(y < 0.5, 
                     tf.where(x < 0.4, k1, k2), 
                     k3)
        return k

# Cast x, y, k to float32
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    k = tf.cast(k, dtype=tf.float32)

    # Create a grid of (x, y) points
    X, Y = tf.meshgrid(x, y)
    X = tf.reshape(X, [-1, 1])
    Y = tf.reshape(Y, [-1, 1])
    xy = tf.concat([X, Y], axis=1)

    # Use persistent=True to allow multiple gradient computations
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        u = model(xy)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            u = model(xy)
        u_x = tape2.gradient(u, xy)[:, 0]
        u_y = tape2.gradient(u, xy)[:, 1]
        del tape2  # Deleting the inner tape

    u_xx = tape.gradient(u_x, xy)[:, 0]
    u_yy = tape.gradient(u_y, xy)[:, 1]
    del tape  # Deleting the outer tape

    # Define the new source term g(x, y)
    g = -6 * X * Y**2 * (X**2 + 2 * Y**2) + 5 * X**3 * Y**4

    # Get the wave number k for each point
    k = wave_number(X, Y)

    # Modified Helmholtz equation
    f = -(u_xx + u_yy) - k**2 * u - g

    # Boundary conditions loss
    bc_loss = apply_2D_boundary_conditions(model, x, y)

    # Total loss is the sum of f_loss and bc_loss
    return tf.reduce_mean(tf.square(f)) + bc_loss    
    




'''
# PINNs loss
def PINNs_loss_2D(model, x, y, k):
    
    # Cast x, y, k to float32
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    k = tf.cast(k, dtype=tf.float32)

    # Create a grid of (x, y) points
    X, Y = tf.meshgrid(x, y)
    X = tf.reshape(X, [-1, 1])
    Y = tf.reshape(Y, [-1, 1])
    xy = tf.concat([X, Y], axis=1)

    # Use persistent=True to allow multiple gradient computations
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        u = model(xy)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            u = model(xy)
        u_x = tape2.gradient(u, xy)[:, 0]
        u_y = tape2.gradient(u, xy)[:, 1]
        del tape2  # Deleting the inner tape

    u_xx = tape.gradient(u_x, xy)[:, 0]
    u_yy = tape.gradient(u_y, xy)[:, 1]
    del tape  # Deleting the outer tape

    # Define the new source term g(x, y)
    g = -6 * X * Y**2 * (X**2 + 2 * Y**2) + 5 * X**3 * Y**4

    # Modified Helmholtz equation
    f = -(u_xx + u_yy) - k**2 * u - g

    # Boundary conditions loss
    bc_loss = apply_2D_boundary_conditions(model, x, y)

    # Total loss is the sum of f_loss and bc_loss
    return tf.reduce_mean(tf.square(f)) + bc_loss
'''



'''

# Helmholtz 2D with g (x), y =δ (x −epsilon_1 , y −epsilon_1)_
    # PINNS loss

def PINNs_loss_2D(model, x, y, k, epsilon1, epsilon2):


# Create a grid of (x, y) points


    # Cast x, y, k, epsilon1, and epsilon2 to float32
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    k = tf.cast(k, dtype=tf.float32)
    epsilon1 = tf.cast(epsilon1, dtype=tf.float32)
    epsilon2 = tf.cast(epsilon2, dtype=tf.float32)

    # Create a grid of (x, y) points
    X, Y = tf.meshgrid(x, y)
    X = tf.reshape(X, [-1, 1])
    Y = tf.reshape(Y, [-1, 1])
    xy = tf.concat([X, Y], axis=1)

    # Use persistent=True to allow multiple gradient computations
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        u = model(xy)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            u = model(xy)
        u_x = tape2.gradient(u, xy)[:, 0]
        u_y = tape2.gradient(u, xy)[:, 1]
        del tape2

    u_xx = tape.gradient(u_x, xy)[:, 0]
    u_yy = tape.gradient(u_y, xy)[:, 1]
    del tape

   # Define an approximation to the Dirac delta function
    # as a narrow Gaussian ('sigma' is the width of the Gaussian)
    sigma = 0.01  # Width of the Gaussian, this is a small number to approximate the Dirac delta
    g = tf.exp(-((X - epsilon1)**2 + (Y - epsilon2)**2) / (2 * sigma**2))
    g = g / (2 * np.pi * sigma**2)  # Normalize the Gaussian
    

    # Modified Helmholtz equation
    f = -(u_xx + u_yy) - k**2 * u - g

    # Boundary conditions loss
    bc_loss = apply_2D_boundary_conditions(model, x, y)




    return tf.reduce_mean(tf.square(f)) + bc_loss

'''
# Data Preparation



x_values = np.linspace(0, 1, 100).astype(np.float32)
y_values = np.linspace(0, 1, 100).astype(np.float32)
K_values = np.array([10.0, 20.0, 30.0], dtype=np.float32)

X, Y = np.meshgrid(x_values, y_values)
inputs = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
targets = np.zeros((inputs.shape[0], 1), dtype=np.float32)
'''
#Model Creation and Compilation
model = create_PINN_model_2D([50, 100, 100, 50], 'tanh')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=lambda _, u_pred: PINNs_loss_2D(model, x_values, y_values, K_values[0], epsilon1, epsilon2))
'''


# Model Creation and Compilation
model = create_PINN_model_2D([50, 100, 100, 50], 'tanh')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=lambda _, u_pred: PINNs_loss_2D(model, x_values, y_values, K_values[0]))


# Training the model
print("Training the model...")

# Define the number of steps you want per epoch
#steps_per_epoch = 1  # This will only process 1 batches per epoch

# Training the model with limited steps per epoch
#history = model.fit(inputs, targets, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)
history = model.fit(inputs, targets, epochs=epochs, verbose=1)
#history = model.fit(inputs, targets, epochs=epochs, verbose=1)
print("Training completed!")

# After training, plot the learning curve
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Plot the training loss
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()



# Define the analytical solution function
def analytical_solution(x, y):
    return x**3 * y**4

# After training is completed, evaluate the model on the grid and compare with analytical solution
X_eval, Y_eval = np.meshgrid(x_values, y_values)
eval_points = np.stack([X_eval.ravel(), Y_eval.ravel()], axis=-1).astype(np.float32)

# Predict the solution over the entire grid
u_pred = model.predict(eval_points).reshape(X_eval.shape)

# Calculate the analytical solution over the grid
u_analytical = analytical_solution(X_eval, Y_eval)

# Plotting
fig = plt.figure(figsize=(12, 5))

# Predicted Solution
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X_eval, Y_eval, u_pred, cmap='viridis', edgecolor='none')
ax1.set_title('Predicted Solution (PINN)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x, y)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Analytical Solution
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X_eval, Y_eval, u_analytical, cmap='viridis', edgecolor='none')
ax2.set_title('Analytical Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u(x, y)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)



# Calculate the difference between the predicted and analytical solutions
difference = u_pred - u_analytical

# Plotting the difference
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
error_surf = ax.plot_surface(X_eval, Y_eval, difference, cmap='hot', edgecolor='none')
ax.set_title('Difference between Predicted (PINN) and Analytical Solutions')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Difference u(x, y)')
fig.colorbar(error_surf, shrink=0.5, aspect=5)

plt.show()


# Convert analytical solution to a TensorFlow tensor for consistency
u_analytical_tf = tf.convert_to_tensor(u_analytical, dtype=tf.float32)

# Convert predicted solution to a TensorFlow tensor
u_pred_tf = tf.convert_to_tensor(u_pred, dtype=tf.float32)

# Calculate Mean Squared Error
mse = tf.reduce_mean(tf.square(u_pred_tf - u_analytical_tf))

# Calculate Mean Absolute Error
mae = tf.reduce_mean(tf.abs(u_pred_tf - u_analytical_tf))

# Print the errors
print(f"Mean Squared Error (MSE) between PINN and analytical solution: {mse.numpy()}")
print(f"Mean Absolute Error (MAE) between PINN and analytical solution: {mae.numpy()}")


# Overlay the heterogeneity regions
# Define the boundaries of the regions (adjust these as per your domain setup)
x_boundary_1 = 0.4
y_boundary = 0.5
x_boundary_2 = 1.0

# Plot the boundaries
ax.plot([x_boundary_1, x_boundary_1], [0, y_boundary], [np.min(u_pred), np.max(u_pred)], 'k--', lw=2)
ax.plot([0, x_boundary_2], [y_boundary, y_boundary], [np.min(u_pred), np.max(u_pred)], 'k--', lw=2)

plt.show()




'''

time-dependent: Helmholtz Equation

# Define the neural network model for PINNs with an additional time input
def create_PINN_model_2D_time(layers, activation_function):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(3,)))  # Input layer for 2D problem and time (x, y, t)
    for layer in layers:
        model.add(tf.keras.layers.Dense(layer, activation=activation_function))
    model.add(tf.keras.layers.Dense(1, activation=None))  # Output layer
    return model

# Adjust the PINNs loss function to include time dependency
def PINNs_loss_2D_time(model, x, y, t, c):
    # ... [rest of the code to set up the computation graph]

    # Compute the gradients with respect to x, y, and t
    # ... [rest of the code to compute spatial derivatives]

    # Compute the second temporal derivative
    with tf.GradientTape() as tape2:
        tape2.watch(t)
        u_t = tape.gradient(u, t)
    u_tt = tape2.gradient(u_t, t)

    # Wave equation
    f = u_tt - c**2 * (u_xx + u_yy) - source_term(x, y, t)

    # Boundary and initial conditions loss
    bc_loss = apply_boundary_conditions(model, x, y, t)

    # Total loss
    return tf.reduce_mean(tf.square(f)) + bc_loss

# Data Preparation including time
t_values = np.linspace(0, T, num_timesteps).astype(np.float32)  # T is the total time, num_timesteps is the number of time points
x_values, y_values, t_values = np.meshgrid(x_values, y_values, t_values)
inputs = np.stack([x_values.ravel(), y_values.ravel(), t_values.ravel()], axis=-1).astype(np.float32)

# Model Creation and Compilation
model = create_PINN_model_2D_time([50, 100, 100, 50], 'tanh')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=lambda _, u_pred: PINNs_loss_2D_time(model, x_values, y_values, t_values, wave_speed))

# Training the model with time-dependent data
history = model.fit(inputs, targets, epochs=epochs, verbose=1)

# After training, predict the solution over the entire space-time grid
predicted_solution = model.predict(inputs).reshape(x_values.shape)


'''