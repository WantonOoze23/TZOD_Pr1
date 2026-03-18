import pandas as pd
import numpy as np
import time

num_rows = 100_000_000
print(f"Генерація датасету на {num_rows} рядків...")

dates = pd.date_range("2000-01-01", periods=num_rows, freq="D")
revenue = np.random.uniform(100, 5000, num_rows)

df = pd.DataFrame({
    "date": dates,
    "revenue": revenue
})

window = 7

print("\nОбчислення через Pandas...")
start_time = time.time()
df['rolling_mean_pd'] = df['revenue'].rolling(window=window).mean()
pandas_time = time.time() - start_time
print(f"Час виконання Pandas: {pandas_time:.4f} секунд")

print("\nОбчислення через NumPy (cumsum)...")
start_time = time.time()

arr = df['revenue'].values

cumsum_arr = np.cumsum(arr, dtype=float)

cumsum_arr[window:] = cumsum_arr[window:] - cumsum_arr[:-window]

numpy_rolling_mean = cumsum_arr[window - 1:] / window

numpy_result_full = np.empty_like(arr)
numpy_result_full[:] = np.nan
numpy_result_full[window - 1:] = numpy_rolling_mean

df['rolling_mean_np'] = numpy_result_full
numpy_time = time.time() - start_time
print(f"Час виконання NumPy: {numpy_time:.4f} секунд")

valid_pd = df['rolling_mean_pd'].dropna().values
valid_np = df['rolling_mean_np'][window - 1:]

max_difference = np.max(np.abs(valid_pd - valid_np))
print(f"\nМаксимальна абсолютна різниця між методами: {max_difference}")