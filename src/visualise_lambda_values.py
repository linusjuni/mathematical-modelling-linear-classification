import numpy as np
import matplotlib.pyplot as plt

lambda_values = np.concatenate([
        #np.logspace(-5, -3, 5),
        #np.linspace(0.001, 0.1, 10), 
        #np.linspace(0.1, 1, 5),
        np.linspace(1, 10, 5),
        #np.linspace(10, 1000, 5)
    ])
lambda_values = np.unique(lambda_values.round(8))

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, 'o-', label='Lambda Values')
plt.xlabel('Index')
plt.ylabel('Lambda Value')
plt.title('Plot of Lambda Values')
plt.legend()
plt.grid(True)
plt.show()

print("Lambda values:", lambda_values)
print("max lambda value:", lambda_values.max())
