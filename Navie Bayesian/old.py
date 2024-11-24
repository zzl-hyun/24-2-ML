import csv
import argparse
from collections import defaultdict
from math import prod

class NaiveBayes:
    def __init__(self):
        self.label_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total_samples = 0

    def train(self, data):
        """
        Train the Naive Bayes model with training data.
        """
        for row in data:
            label = row[-1]
            self.label_counts[label] += 1
            self.total_samples += 1
            for i, feature in enumerate(row[:-1]):
                self.feature_counts[i][(feature, label)] += 1

    def predict(self, features):
        """
        Predict the probabilities for the input features.
        """
        results = {}
        for label in self.label_counts:
            # Calculate prior probability
            prior = self.label_counts[label] / self.total_samples
            likelihood = prod(
                (self.feature_counts[i].get((feature, label), 0) + 1) /
                (self.label_counts[label] + len(self.feature_counts[i]))
                for i, feature in enumerate(features)
            )
            results[label] = prior * likelihood
        return results

    @staticmethod
    def format_output(features, probabilities):
        """
        Format the output as required, including the input features.
        """
        total = sum(probabilities.values())
        ratios = {k: v / total for k, v in probabilities.items()}
        print(f"{', '.join(features)}")
        for label, ratio in ratios.items():
            print(f"{label} ({ratio:.5f})")
        print()  # Add a blank line for better readability

def load_csv(filepath, has_label=True):
    """
    Load data from a CSV file.
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        data = [row for row in reader]
    if has_label:
        return data
    else:
        return [row for row in data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Binary Classifier")
    parser.add_argument("--train", required=True, help="Path to training CSV file")
    parser.add_argument("--test", required=True, help="Path to testing CSV file")
    args = parser.parse_args()

    # Load training data
    training_data = load_csv(args.train)
    nb = NaiveBayes()
    nb.train(training_data)

    # Load testing data
    testing_data = load_csv(args.test, has_label=False)
    
    # Make predictions and output results
    for features in testing_data:
        probabilities = nb.predict(features)
        nb.format_output(features, probabilities)