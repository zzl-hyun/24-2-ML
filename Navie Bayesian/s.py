from sklearn.utils import resample
import pandas as pd


data = pd.read_csv('extended_imbalanced_playtennis.csv')

# Separate majority and minority classes
majority_class = data[data['Play Tennis'] == 'Yes']
minority_class = data[data['Play Tennis'] == 'No']

# Perform Subsampling on the majority class
majority_downsampled = resample(
    majority_class,
    replace=False,  # sample without replacement
    n_samples=len(minority_class),  # match minority class count
    random_state=42
)

# Perform Oversampling on the minority class
minority_oversampled = resample(
    minority_class,
    replace=True,  # sample with replacement
    n_samples=len(majority_class),  # match majority class count
    random_state=42
)

# Combine the downsampled majority class and oversampled minority class
balanced_data = pd.concat([majority_downsampled, minority_oversampled])

# Shuffle the resulting dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV file
output_path = 'balanced_playtennis2.csv'
balanced_data.to_csv(output_path, index=False)

output_path
