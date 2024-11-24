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
    label_counts = Counter()    # 레이블 수 계산
    for row in data:
        label_counts[row[-1]] += 1

    entropy = 0
    for count in label_counts.values():
        p = count / len(data)   # 레이블의 비율
        entropy -= p * math.log2(p) 
        # print(entropy)
    return entropy

def information_gain(data, feature_index):
    total_entropy = entropy(data)   # 전체 엔트로피
    feature_values = []
    for row in data:
        feature_values.append(row[feature_index])   # 특정 feature의 값들

    value_counts = Counter(feature_values)  # 특정 feature의 값들의 수
    feature_entropy = 0
    
    for value, count in value_counts.items(): # 특정 feature의 값들에 대한 엔트로피  
        subset = [] 
        for row in data: # 특정 feature의 값들에 대한 데이터
            # print(row[0])
            if row[feature_index] == value: 
                subset.append(row)  
        # 특정 feature의 값들에 대한 엔트로피   
        feature_entropy += (count / len(data)) * entropy(subset) 

    # print(f"Gain({header[feature_index]}): {total_entropy - feature_entropy:.3f}")
    return total_entropy - feature_entropy

def select_best_feature(data, remaining_features):
    gains = {}
    for i, feature in enumerate(header[:-1]):
        if feature in remaining_features:
            gains[feature] = information_gain(data, i)
            
    #print(gains)
    best_feature = max(gains, key=gains.get)
    best_feature_index = header.index(best_feature)
    # print(f"Best Feature: {best_feature}")
    
    return best_feature, best_feature_index

def split_data(data, feature_index):
    groups = {}
    for row in data:
        feature_value = row[feature_index]
        # print(row[feature_index])
        if feature_value not in groups:
            groups[feature_value] = []
            
        groups[feature_value].append(row)
    # print(groups)
    return groups

def decision_tree_train(data, features):
    labels = []
    for row in data:
        labels.append(row[-1])

    most_common_label = Counter(labels).most_common(1)[0][0] # 가장 많은 레이블 -> 예측값
    
    if len(set(labels)) == 1:
        return Leaf(most_common_label) # 모든 레이블이 같을 때
    if not features:
        return Leaf(most_common_label) # 사용할 feature가 없을 때

    best_feature, best_feature_index = select_best_feature(data, features)
    groups = split_data(data, best_feature_index)   # best feature로 데이터 분할
    remaining_features = features - {best_feature} # 사용한 feature 제외 (나머지 feature)

    branch = {}
    for value, subset in groups.items():
        branch[value] = decision_tree_train(subset, remaining_features) # 재귀적으로 트리 생성

    return Node(best_feature, branch)
            
def print_if_then(node, depth=0):
    if isinstance(node, Leaf):
        return "\t" * depth + f"then Play Tennis = {node.value}"
    else:
        results = []
        for value, branch in node.branch.items():
            str = "\t" * depth + f"if {node.feature} is {value}"
            results.append(str + "\n" + print_if_then(branch, depth + 1))
            
        return '\n'.join(results)
           
if __name__ == "__main__":
    data = read_csv(sys.argv[1])
    header = data[0]
    rows = data[1:]
    # print(header[:-1])
    features = set(header[:-1])

    tree = decision_tree_train(rows, features)

    print(print_if_then(tree))
    