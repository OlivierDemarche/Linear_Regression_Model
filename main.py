import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# -------------------------   GENERATION -------------------------
# -------------------  ETAPES 1 : GENERER DONNEES ----------------
x, y = make_regression(n_samples=100, n_features=1, noise=10)  # Création du DATASET

plt.figure()  # Afficher les données
plt.title("Data Set")
plt.scatter(x, y)
plt.show()

print(x.shape)  # Modifier la forme de notre matrice y
y = y.reshape(y.shape[0], 1)
print(y.shape)
# ---------------------------- CREATION MATRICE X ---------------------
X = np.hstack((x, np.ones(x.shape)))
print(X.shape)
print(X)
# ----------------------------  INITIALISATION THETA ---------------------
theta = np.random.randn(2, 1)
print(theta.shape)
print(theta)


# -----------------------------  FONCTIONS DU MODELE ------------------------
def model(X, theta):
    return X.dot(theta)


def cost_function(X, y, theta):
    m = len(y)
    return 1 / (2 * m) * np.sum((model(X, theta) - y) ** 2)


def grad(X, y, theta):
    m = len(y)
    return 1 / m * X.T.dot(model(X, theta) - y)


def gradient_descent(X, y, theta, learning_rate, n_iteration):
    cost_history = np.zeros(n_iteration)
    for i in range(0, n_iteration):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history


def coef_determination(y, prediction):
    u = ((y - prediction) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - u / v


# ------------------------------- RESULTATS -----------------------------
# THETA
theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, n_iteration=1000)
print(theta_final)

# AFFICHAGE DU MODELE
prediction = model(X, theta_final)
plt.scatter(x, y)
plt.plot(x, prediction, c='r')
plt.show()

# AFFICHAGE EVOLUTION DU MODELE
plt.plot(range(1000), cost_history)
plt.show()
# Print the coefficient of determination (rsquared)
print(coef_determination(y, prediction))
