import csv
import argparse
from collections import defaultdict
from math import prod


class NaiveBayes:
    def __init__(self):
        self.label_counts = defaultdict(int)  # Label별 샘플 수
        self.feature_counts = defaultdict(lambda: defaultdict(int))  # Feature별 조건부 확률 계산용
        self.feature_values = defaultdict(set)  # 각 특징별 가능한 값 저장
        self.total_samples = 0  # 전체 샘플 수

    def train(self, data):
        for row in data:
            label = row[-1]  # 마지막 열이 라벨 값
            self.label_counts[label] += 1
            self.total_samples += 1
            for i, feature in enumerate(row[:-1]):  # 마지막 열 제외
                self.feature_counts[i][(feature, label)] += 1 # 카운트
                self.feature_values[i].add(feature)  # 각 특징의 가능한 값을 저장

    def predict(self, features):
        probabilities = {}

        for label in self.label_counts:
            # 사전 확률 계산
            prior = self.label_counts[label] / self.total_samples  # P(label)
            likelihood = 1
            for i in range(len(features)):
                feature = features[i]   

                nk = self.feature_counts[i].get((feature, label), 0) # 특정 레이블 v에서 전체 샘플 수                              
                n = self.label_counts[label] # 레이블 v에서 전체 샘플 수                           
                m = len(self.feature_values[i]) # 등가 샘플 크기 (m = V)
                p = 1 / m   # 사전 확률 p = 1 / V (균등 분포 가정)

                # 스무딩 적용:  (nk + m * p) / (n + m)
                # Likelihood에 곱해줌
                likelihood *= (nk + m * p) / (n + m)

            # 최종 확률 계산: Prior * Likelihood
            probabilities[label] = prior * likelihood
        return probabilities

def read_csv(filepath):
    data =[]
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    # print(data[1:])
    return data[1:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options --train --test")
    parser.add_argument("--train", required=True, help="training file path")
    parser.add_argument("--test", required=True, help="testing file path")
    args = parser.parse_args()

    # 학습
    training_data = read_csv(args.train)
    
    nb = NaiveBayes()
    nb.train(training_data)

    # 예측
    testing_data = read_csv(args.test)
    for features in testing_data:
        probabilities = nb.predict(features)

        pos = probabilities.get("Yes", 0)
        neg = probabilities.get("No", 0)
        ratio = round(pos / neg, 5)

        print(f"{', '.join(features)}")
        print(f"Yes ({pos:.5f}) No ({neg:.5f}) Ratio ({ratio})\n")

