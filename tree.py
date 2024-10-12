import csv
import sys
import math
from collections import Counter

# 결정 기준
class Node:
    def __init__(self, feature, branch):
        self.feature = feature
        self.branch = branch

# 예측 값
class Leaf:
    def __init__(self, value):
        self.value = value


def read_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = list(reader)
    
    return data

# 엔트로피 계산 log2로 계산!!
def entropy(data):
    label_counts = Counter()
    for row in data:
        label_counts[row[-1]] += 1

    entropy = 0
    for count in label_counts.values():
        p = count / len(data)
        entropy -= p * math.log2(p) 
        # print(entropy)
    return entropy


def information_gain(data, feature_index):
    total_entropy = entropy(data)
    feature_values = []

    for row in data:
        feature_values.append(row[feature_index])

    value_counts = Counter(feature_values)
    feature_entropy = 0
    
    for value, count in value_counts.items():
        subset = []
        for row in data:
            # print(row[0])
            if row[feature_index] == value:
                subset.append(row)
        feature_entropy += (count / len(data)) * entropy(subset)

    # print(f"Gain({header[feature_index]}): {total_entropy - feature_entropy:.3f}")
    return total_entropy - feature_entropy

def select_best_feature(data, remaining_features):
    gains = {}

    for i, feature in enumerate(header[:-1]):
        if feature in remaining_features:
            gains[feature] = information_gain(data, i)
            
    best_feature = max(gains, key=gains.get)
    best_feature_index = header.index(best_feature)
    # print(f"Best Feature: {best_feature}")
    
    return best_feature, best_feature_index

def split_data(data, feature_index):
    groups = {}

    for row in data:
        feature_value = row[feature_index]
        if feature_value not in groups:
            groups[feature_value] = []
            
        groups[feature_value].append(row)
    # print(data)
    # print(groups)
    return groups

def decision_tree_train(data, features):
    labels = []
    
    for row in data:
        labels.append(row[-1])

    label_counts = Counter(labels)
    most_common_label = label_counts.most_common(1)[0][0]
    
    if len(set(labels)) == 1:
        return Leaf(most_common_label)
    if not features:
        return Leaf(most_common_label)

    best_feature, best_feature_index = select_best_feature(data, features)
    remaining_features = features - {best_feature}
    groups = split_data(data, best_feature_index)
    branch = {}

    for value, subset in groups.items():
        branch[value] = decision_tree_train(subset, remaining_features)
    return Node(best_feature, branch)
            
        
def print_if_then(node, depth=0):
    if isinstance(node, Leaf):
        return "\t" * depth + f"then Play Tennis = {node.value}"
    else:
        results = []
        for value, branch in node.branch.items():
            condition = "\t" * depth + f"if {node.feature} is {value}"
            subtree = print_if_then(branch, depth + 1)
            results.append(condition + "\n" + subtree)
            
        return '\n'.join(results)
           
if __name__ == "__main__":
    data = read_csv(sys.argv[1])
    header = data[0]
    rows = data[1:]
    # print(header[:-1])
    features = set(header[:-1])

    tree = decision_tree_train(rows, features)

    print(print_if_then(tree))
    