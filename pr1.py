import numpy as np

np.random.seed(42)
n_samples = 200

price = np.random.uniform(10, 100, n_samples)
quantity = np.random.randint(1, 10, n_samples)
discount = np.random.uniform(0, 0.3, n_samples)

noise = np.random.normal(0, 5, n_samples)
total = 10 + 2 * price + 15 * quantity - 30 * discount + noise

def fit_linear_regression(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]

    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta, X_b


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


X_multi = np.c_[price, quantity, discount]
theta_multi, X_b_multi = fit_linear_regression(X_multi, total)

total_pred_multi = X_b_multi.dot(theta_multi)
mse_multi = mean_squared_error(total, total_pred_multi)

print("Модель 1: total ~ price + quantity + discount")
print(f"Коефіцієнти (b0, price, quantity, discount): {theta_multi.round(2)}")
print(f"MSE (Множинна регресія): {mse_multi:.2f}\n")

X_simple = price.reshape(-1, 1)
theta_simple, X_b_simple = fit_linear_regression(X_simple, total)

total_pred_simple = X_b_simple.dot(theta_simple)
mse_simple = mean_squared_error(total, total_pred_simple)

print("Модель 2: total ~ price")
print(f"Коефіцієнти (b0, price): {theta_simple.round(2)}")
print(f"MSE (Проста регресія): {mse_simple:.2f}\n")

print("Висновок")
if mse_multi < mse_simple:
    print(f"Множинна модель краща. Помилка зменшилась на {mse_simple - mse_multi:.2f}")
else:
    print("Проста модель показала кращий або однаковий результат.")