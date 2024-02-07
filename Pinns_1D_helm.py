import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network model for PINNs
def create_PINN_model(layers, activation_function):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))  # Input layer for one-dimensional problem
    for layer in layers:
        model.add(tf.keras.layers.Dense(layer, activation=activation_function))
    model.add(tf.keras.layers.Dense(1, activation=None))  # Output layer
    return model

# Define the boundary conditions for the Helmholtz equation
def apply_dirichlet_boundary_conditions(u, x):
    bc0 = u[0] - 1
    h = x[1] - x[0]
    bc1 = (u[-1] - u[-2]) / h
    return bc0**2 + bc1**2

# Define the loss function for PINNs
def PINNs_loss(model, x, k):
    with tf.GradientTape() as tape:
        tape.watch(x)
        u = model(x)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            u = model(x)
        u_x = tape2.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    
    f = u_xx + k**2 * u
    bc_loss = apply_dirichlet_boundary_conditions(u, x)

    return tf.reduce_mean(tf.square(f)) + bc_loss

# Define parameters
k_values = [2]  # Different values for k
layers = [50, 100, 100, 50] # Define the layers
activation_function = 'tanh'
learning_rate = 0.001
epochs = 10000

# Prepare the data
x_values = np.linspace(0, 1, 100).reshape(-1, 1)
x_tf = tf.convert_to_tensor(x_values, dtype=tf.float32)

for k in k_values:
    # Create and compile the model
    model = create_PINN_model(layers, activation_function)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=lambda y, u_pred: PINNs_loss(model, x_tf, k))

    # Train the model
    print(f"Training model with k = {k}")
    history = model.fit(x_values, np.zeros_like(x_values), epochs=epochs, verbose=1)

    # Predict the solution
    u_pred = model.predict(x_values).flatten()

    # Analytical solution
    u_analytical = np.cos(k * x_values) + np.tan(k) * np.sin(k * x_values)

    # Convert analytical solution to a TensorFlow tensor for consistency
    u_analytical_tf = tf.convert_to_tensor(u_analytical, dtype=tf.float32)

    # Convert predicted solution to a TensorFlow tensor
    u_pred_tf = tf.convert_to_tensor(u_pred, dtype=tf.float32)

    # Calculate Mean Squared Error
    mse = tf.reduce_mean(tf.square(u_pred_tf - u_analytical_tf))

    # Calculate Mean Absolute Error
    mae = tf.reduce_mean(tf.abs(u_pred_tf - u_analytical_tf))

    # After training, evaluate the model predictions and calculate the residuals
    u_pred = model.predict(x_values).flatten()
    u_analytical = np.cos(k * x_values) + np.tan(k) * np.sin(k * x_values)
    
    # Calculate residuals
    residuals = u_pred - u_analytical.flatten()

    # Print the errors
    print(f"Mean Squared Error (MSE) between PINN and analytical solution: {mse.numpy()}")
    print(f"Mean Absolute Error (MAE) between PINN and analytical solution: {mae.numpy()}")

    # Plot the training loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'])
    plt.title(f'Model Loss for k={k}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    # Plot the residuals vs epochs
    residuals = u_pred - u_analytical.flatten()
    plt.figure(figsize=(10, 4))
    plt.plot(residuals)
    plt.title(f'Residuals vs. Epochs for k={k}')
    plt.ylabel('Residual')
    plt.xlabel('Epoch')
    plt.show()

# Comparison of PINN Predicted Solution and Analytical Solution for k
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, u_pred, label='Predicted Solution (PINN)')
    plt.plot(x_values, u_analytical, '--', label='Analytical Solution')
    plt.title(f'Comparison of PINN Predicted Solution and Analytical Solution for k={k}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.show()

# Plot the difference between predicted and analytical solutions
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, residuals, label='Residuals')
    plt.title(f'Residuals (Predicted - Analytical) for k={k}')
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.legend()
    plt.show()


