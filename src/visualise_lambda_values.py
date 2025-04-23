import numpy as np
import matplotlib.pyplot as plt
from part_1 import lambda_values

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
