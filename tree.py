import csv
import sys
import math
from collections import Counter

class Node:
    def __init__(self, feature, branch):
        self.feature = feature
        self.branch = branch
class Leaf:
    def __init__(self, value):
        self.value = value

def read_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = list(reader)

    return data

def entropy(data_subset):
    label_counts = Counter(row[-1] for row in data_subset)
    total_samples = len(data_subset)
    ent = 0
    for count in label_counts.values():
        p = count / total_samples
        ent -= p * math.log2(p) 
        
    return ent

def information_gain(data, feature_index):
    total_entropy = entropy(data)
    
    feature_values = [row[feature_index] for row in data]
    value_counts = Counter(feature_values)
    total_samples = len(data)
    
    feature_entropy = 0
    for value, count in value_counts.items():
        subset = []
        for row in data:
            # print(row[0])
            if row[feature_index] == value:
                subset.append(row)
        feature_entropy += (count / total_samples) * entropy(subset)

    print(f"Gain({header[feature_index]}): {total_entropy - feature_entropy:.3f}")
    return total_entropy - feature_entropy

def decision_tree_train(data, remaining_features):
    # Count the labels in the data
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    most_common_label = label_counts.most_common(1)[0][0]  # Default guess is the most frequent label
    # print(most_common_label)
    # Base cases
    if len(set(labels)) == 1:  # If all labels are the same
        return Leaf(most_common_label)
    
    if not remaining_features:  # If no features left to split
        return Leaf(most_common_label)
    
    # Calculate Information Gain for each remaining feature
    gains = {}
    for i, feature in enumerate(header[:-1]):
        if feature in remaining_features:
            gains[feature] = information_gain(data, i)
    
    # Select the feature with the highest Information Gain
    best_feature = max(gains, key=gains.get)
    best_feature_index = header.index(best_feature)
    print(f"Best Feature: {best_feature}\n")
    
    # print(f"gains: {gains}, best_feature: {best_feature}")
    
    # Partition the data based on the best feature's values
    partitions = {}
    for row in data:
        feature_value = row[best_feature_index]
        if feature_value not in partitions:
            partitions[feature_value] = []
        partitions[feature_value].append(row)
        
    # Recursively build the branch
    branch = {}
    remaining_features = remaining_features - {best_feature}
    for value, subset in partitions.items():
        branch[value] = decision_tree_train(subset, remaining_features)
    
    return Node(best_feature, branch)

# Function to visualize the tree
def print_tree(node, depth=0):
    indent = "  " * depth
    if isinstance(node, Leaf):
        print("  "*depth, f"then Play Tennis = {node.value}")
    else:
        print(f"{indent}Node: {node.feature}")
        for value, branch in node.branch.items():
            print(f"{indent}  {value}:")
            print_tree(branch, depth + 2)
            
def print_if_then_tree(node):
    if isinstance(node, Leaf):
        return f"then Play Tennis = {node.value}"
    else:
        results = []
        for value, branch in node.branch.items():
            condition = f"if {node.feature} is {value}, "
            results.append(condition + print_if_then_tree(branch))
        return '\n'.join(results)
           
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python .py .csv")
        sys.exit(1)
    
    data = read_csv(sys.argv[1])
    header = data[0]
    data_rows = data[1:]

    # # Calculate entropy of the whole dataset
    # total_entropy = entropy(data_rows)
    # print(f"Entropy(D): {total_entropy}")
    # gains = {}
    # for i in range(len(header) - 1):  # Exclude the target label column
    #     gains[header[i]] = information_gain(data_rows, i)
    
    #print(total_entropy, gains)
    
    # Initial call to the decision tree function
    remaining_features = set(header[:-1])  # Exclude target label "Play Tennis"
    tree = decision_tree_train(data_rows, remaining_features)
    print_tree(tree)
    print(print_if_then_tree(tree))
    