import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_data

train_path = "/Users/linus.juni/Documents/Personal/mathematical-modelling-linear-classification/data/train"
test_path = "/Users/linus.juni/Documents/Personal/mathematical-modelling-linear-classification/data/test"

print("Loading training data...")
X_train, y_train = load_data(train_path)
print("Loading test data...")
X_test, y_test = load_data(test_path)

# Check for class imbalance
train_positive = sum(y_train)
train_negative = len(y_train) - train_positive
test_positive = sum(y_test)
test_negative = len(y_test) - test_positive

print(f"Training set: {train_positive} positive, {train_negative} negative")
print(f"Test set: {test_positive} positive, {test_negative} negative")

labels = ['Train Positive', 'Train Negative', 'Test Positive', 'Test Negative']
values = [train_positive, train_negative, test_positive, test_negative]

sns.set_theme(style="whitegrid")
colors = ['#87CEEB', '#FFB6C1', '#87CEEB', '#FFB6C1']
labels = ['Train Positive', 'Train Negative', 'Test Positive', 'Test Negative']
values = [train_positive, train_negative, test_positive, test_negative]

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=values, palette=colors)
plt.title('Class Distribution in Train and Test Sets', fontsize=14)
plt.ylabel('Number of Samples', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()