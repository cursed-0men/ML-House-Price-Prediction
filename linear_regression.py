import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df[['sqft_living','bedrooms','bathrooms','price']].dropna()

    x1 = df['sqft_living'].values
    x2 = df['bedrooms'].values
    x3 = df['bathrooms'].values
    y = df['price'].values

    # Standardize
    x1 = (x1 - np.mean(x1)) / np.std(x1)
    x2 = (x2 - np.mean(x2)) / np.std(x2)
    x3 = (x3 - np.mean(x3)) / np.std(x3)
    y = (y - np.mean(y)) / np.std(y)

    x = np.column_stack((np.ones_like(x1), x1, x2, x3))
    return x, y, x1, x2, x3, df

# Train the model using gradient descent
def train_gradient_descent(x, y, lr=0.01, steps=1000):
    b = np.zeros(x.shape[1])
    n = len(x)
    residual_list = []

    for i in range(steps):
        y_pred = x @ b
        residual = y - y_pred
        loss_mse = (1/n) * np.sum((residual)**2)
        residual_list.append(loss_mse)
        gradient = (-2/n) * (x.T @ residual)
        b = b - (lr * gradient)
    return b, residual_list

def predict_price(sqft_living, bedrooms, bathrooms, b, df):
    x1 = (sqft_living - np.mean(df['sqft_living'])) / np.std(df['sqft_living'])
    x2 = (bedrooms - np.mean(df['bedrooms'])) / np.std(df['bedrooms'])
    x3 = (bathrooms - np.mean(df['bathrooms'])) / np.std(df['bathrooms'])
    x_input = np.array([1, x1, x2, x3])
    y_pred_norm = x_input @ b
    y_pred = (y_pred_norm * np.std(df['price'])) + np.mean(df['price'])
    return y_pred


# Plot loss curve
def plot_loss_curve(residual_list):
    plt.plot(range(len(residual_list)), residual_list, color='green')
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    plt.title("Loss Curve")
    plt.grid(axis='both', linestyle='dashed')
    plt.tight_layout()
    plt.show()

# Plot predicted vs actual prices (denormalized)
def plot_predictions(x, b, df):
    y_pred_final = x @ b
    y_pred_denorm = (y_pred_final * np.std(df['price'])) + np.mean(df['price'])
    y_actual_denorm = df['price'].values

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_actual_denorm, y_pred_denorm, s=60, color='gold', edgecolor='salmon', alpha=0.7, label='Predicted vs Actual')
    min_val = min(y_actual_denorm.min(), y_pred_denorm.min())
    max_val = max(y_actual_denorm.max(), y_pred_denorm.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Ideal Prediction Line')
    ax.set_xlabel("Actual Prices ($)", fontsize=11)
    ax.set_ylabel("Predicted Prices ($)", fontsize=11)
    ax.set_title("Predicted vs Actual Prices (Denormalized)", fontsize=13)
    ax.legend()
    ax.grid(linestyle='dashed', alpha=0.6)
    return fig

# Plot 3D regression plane
def plot_3d_regression(x2, x3, y, x, b):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x2, x3, y, color='blue', s=50, label='Actual Data')
    y_pred = x @ b
    ax.scatter(x2, x3, y_pred, color='red', s=50, label='Predicted Data')

    x2_grid, x3_grid = np.meshgrid(
        np.linspace(min(x2), max(x2), 10),
        np.linspace(min(x3), max(x3), 10)
    )
    x1_fixed = 0
    y_grid = b[0] + b[1]*x1_fixed + b[2]*x2_grid + b[3]*x3_grid

    ax.plot_surface(x2_grid, x3_grid, y_grid, color='yellow', alpha=0.6)
    ax.set_xlabel('Bedrooms', fontsize=10)
    ax.set_ylabel('Bathrooms', fontsize=10)
    ax.set_zlabel('Price (y)', fontsize=10)
    ax.set_title('House Price Prediction – MLR via Gradient Descent', fontsize=12)

    actual_patch = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=6, label='Actual Data')
    pred_patch = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=6, label='Predicted Data')
    plane_patch = mpatches.Patch(color='yellow', alpha=0.6, label='Plane of Regression')
    ax.legend(handles=[actual_patch, pred_patch, plane_patch])

    return fig
