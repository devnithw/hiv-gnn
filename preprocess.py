import os
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Import original CSV file
df = pd.read_csv('HIV.csv')

# Separate majority and minority classes
df_majority = df[df.HIV_active == 0]
df_minority = df[df.HIV_active == 1]

neg_class = df["HIV_active"].value_counts()[0]
pos_class = df["HIV_active"].value_counts()[1]

# Oversampling multiplier
multiplier = int(neg_class/pos_class) - 1

# Manual override multiplier
multiplier = 7

# Samples needed for minority class
n_min_samples = multiplier * len(df_minority)

# Upsample minority class
df_minority_oversampled = resample(df_minority,
                                 replace=True,    # sample with replacement
                                 n_samples=n_min_samples,  # Increase the minority class size by 5
                                 random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
df_oversampled = pd.concat([df_majority, df_minority_oversampled])

# Shuffle dataset before saving
final_df = df_oversampled.sample(frac=1, random_state=42)

# Split data
train_df, test_df = train_test_split(final_df, test_size=0.2)

# Display dataset sizes
print("Negative examples in train data: ", train_df["HIV_active"].value_counts()[0])
print("Posiitve examples in train data: ", train_df["HIV_active"].value_counts()[1])

print("Negative examples in test data: ", test_df["HIV_active"].value_counts()[0])
print("Posiitve examples in test data: ", test_df["HIV_active"].value_counts()[1])

# Create directories
os.makedirs('data/train/raw', exist_ok=True)
os.makedirs('data/train/processed', exist_ok=True)
os.makedirs('data/test/raw', exist_ok=True)
os.makedirs('data/test/processed', exist_ok=True)

# Save the datasets to CSV files
train_df.to_csv('data/train/raw/train.csv', index=False)
test_df.to_csv('data/test/raw/test.csv', index=False)