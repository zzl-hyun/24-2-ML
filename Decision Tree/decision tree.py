import csv
import math
import sys
from collections import Counter, defaultdict

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def calculate_entropy(data):
    label_counts = Counter(row['Play Tennis'] for row in data)
    total_instances = len(data)
    entropy = 0.0
    for count in label_counts.values():
        probability = count / total_instances
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_information_gain(data, feature):
    total_entropy = calculate_entropy(data)
    feature_values = defaultdict(list)
    for row in data:
        feature_values[row[feature]].append(row)
    
    weighted_entropy = 0.0
    total_instances = len(data)
    for subset in feature_values.values():
        subset_entropy = calculate_entropy(subset)
        weighted_entropy += (len(subset) / total_instances) * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

def build_tree(data, features):
    labels = [row['Play Tennis'] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if not features:
        return Counter(labels).most_common(1)[0][0]
    
    best_feature = max(features, key=lambda feature: calculate_information_gain(data, feature))
    tree = {best_feature: {}}
    feature_values = set(row[best_feature] for row in data)
    
    for value in feature_values:
        subset = [row for row in data if row[best_feature] == value]
        subtree = build_tree(subset, [f for f in features if f != best_feature])
        tree[best_feature][value] = subtree
    
    return tree

def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for key, subtree in tree.items():
            for value, subsubtree in subtree.items():
                print(f"{indent}if {key} == {value}:")
                print_tree(subsubtree, indent + "  ")
    else:
        print(f"{indent}then Play Tennis = {tree}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 202312123_ML_hw1.py playtennis.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    data = read_csv(filename)
    features = list(data[0].keys())
    features.remove('Play Tennis')
    
    decision_tree = build_tree(data, features)
    print_tree(decision_tree)