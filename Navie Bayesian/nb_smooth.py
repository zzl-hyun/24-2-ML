import csv
import argparse
from collections import defaultdict
from math import prod


class NaiveBayes:
    def __init__(self):
        self.label_counts = defaultdict(int)  # Label별 샘플 수
        self.feature_counts = defaultdict(lambda: defaultdict(int))  # Feature별 조건부 확률 계산용
        self.total_samples = 0  # 전체 샘플 수
        self.feature_values = defaultdict(set)  # 각 특징별 가능한 값 저장

    def train(self, data):
        for row in data:
            label = row[-1]  # 마지막 열이 라벨 값
            self.label_counts[label] += 1
            self.total_samples += 1
            for i, feature in enumerate(row[:-1]):  # 마지막 열 제외
                self.feature_counts[i][(feature, label)] += 1
                self.feature_values[i].add(feature)  # 각 특징의 가능한 값을 저장

    def predict(self, features):
        probabilities = {}
        for label in self.label_counts:
            # 사전 확률 계산
            prior = self.label_counts[label] / self.total_samples
            
            # 우도 계산 (스무딩 적용)
            likelihood = prod(
                (self.feature_counts[i].get((feature, label), 0) + 1) / (self.label_counts[label] + len(self.feature_values[i]))
                for i, feature in enumerate(features)
            )
            probabilities[label] = prior * likelihood
        return probabilities

    @staticmethod
    def format_output(features, probabilities):
        """
        Format and print the output with probabilities and the ratio.
        """
        yes_prob = probabilities.get("Yes", 0)
        no_prob = probabilities.get("No", 0)

        # Ratio 계산 및 출력
        ratio = round(yes_prob / no_prob, 5) if no_prob > 0 else "Undefined"
        print(f"{', '.join(features)}")
        print(f"Yes ({yes_prob:.5f}) No ({no_prob:.5f})")
        print(f"Ratio = {ratio}\n")


def read_csv(filepath):
    data =[]
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    # print(data[1:])
    return data[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options --train train.csv --test test.csv")
    parser.add_argument("--train", required=True, help="training file path")
    parser.add_argument("--test", required=True, help="testing file path")
    args = parser.parse_args()

    # 훈련 데이터 로드 및 학습
    training_data = read_csv(args.train)
    print(training_data)
    
    nb = NaiveBayes()
    nb.train(training_data)

    # 테스트 데이터 로드 및 예측
    testing_data = read_csv(args.test)
    print(testing_data)

    # Make predictions and output results
    for features in testing_data:
        probabilities = nb.predict(features)
        # print(features)
        nb.format_output(features, probabilities)
