import csv
import sys
from collections import defaultdict
import math

# 데이터를 학습 및 테스트용으로 분리
def load_data(filename):
    try:
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = list(reader)
        if not data or len(data) < 2:
            raise ValueError("CSV 파일이 비었거나 잘못된 형식입니다.")
        header = data[0]
        rows = data[1:]
        for row in rows:
            if len(row) != len(header):
                raise ValueError(f"데이터와 헤더의 열 개수가 일치하지 않습니다: {row}")
        features = header[:-1]  # 마지막 열을 제외한 나머지가 feature
        label_col = header[-1]  # 마지막 열이 label
        return rows, features, label_col
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filename}")
    except Exception as e:
        raise Exception(f"데이터 로드 중 오류 발생: {e}")

# 학습 데이터로 Naive Bayes 모델 생성
def train_naive_bayes(train_data, features, label_col):
    label_counts = defaultdict(int)
    feature_value_counts = {feature: defaultdict(lambda: defaultdict(int)) for feature in features}
    
    for row in train_data:
        label = row[-1]
        label_counts[label] += 1
        for feature, value in zip(features, row[:-1]):
            feature_value_counts[feature][value][label] += 1
            
    return label_counts, feature_value_counts

# Naive Bayes 확률 계산
def calculate_probabilities(test_row, label_counts, feature_value_counts, features, total_samples):
    probabilities = {}
    
    for label, count in label_counts.items():
        log_prob = math.log(count / total_samples)  # P(label)
        for feature, value in zip(features, test_row):
            if value in feature_value_counts[feature]:
                value_count = feature_value_counts[feature][value][label]
                feature_count = sum(feature_value_counts[feature][value].values())
                log_prob += math.log((value_count + 1) / (feature_count + len(label_counts)))  # Laplace smoothing
            else:
                log_prob += math.log(1 / (len(label_counts)))  # Handling unseen features
        probabilities[label] = math.exp(log_prob)
    
    return probabilities

# 모델 평가
def evaluate_naive_bayes(test_data, label_counts, feature_value_counts, features):
    results = []
    total_samples = sum(label_counts.values())
    for test_row in test_data:
        if len(test_row) != len(features):
            raise ValueError(f"테스트 데이터의 피처 개수가 일치하지 않습니다: {test_row}")
        probabilities = calculate_probabilities(test_row, label_counts, feature_value_counts, features, total_samples)
        results.append(probabilities)
    return results

if __name__ == "__main__":

    # 데이터 로드 및 학습
    if len(sys.argv) < 2:
        raise ValueError("CSV 파일 이름을 명령줄 인수로 제공해야 합니다.")
    
    filename = sys.argv[1]
    train_rows, features, label_col = load_data(filename)
    label_counts, feature_value_counts = train_naive_bayes(train_rows, features, label_col)

    # 테스트 데이터 (Play Tennis는 제공되지 않음)
    test_data = [row[:-1] for row in train_rows]

    # 모델 평가
    results = evaluate_naive_bayes(test_data, label_counts, feature_value_counts, features)

    # 결과 출력
    for test_row, probabilities in zip(test_data, results):
        print(", ".join(test_row))  # 입력 데이터 출력
        for label, ratio in probabilities.items():
            print(f"{label} ({ratio:.5f})")
        print()  # 각 테스트 샘플 간 빈 줄 추가
