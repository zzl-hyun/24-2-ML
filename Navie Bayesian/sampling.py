import csv
from collections import Counter
import random

# CSV 파일 읽기
def read_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 헤더 저장
        data = [row for row in reader]  # 데이터 저장
    return header, data

# CSV 파일 쓰기
def write_csv(file_path, header, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
# Subsampling: 다수 클래스 데이터 줄이기
def subsample_data(data, target_index, target_class, target_count):
    filtered_data = [row for row in data if row[target_index] == target_class]
    sampled_data = random.sample(filtered_data, target_count)  # 일부 샘플링
    return sampled_data
# Oversampling: 소수 클래스 데이터 늘리기
def oversample_data(data, target_index, target_class, target_count):
    filtered_data = [row for row in data if row[target_index] == target_class]
    oversampled_data = random.choices(filtered_data, k=target_count)  # 복제
    return oversampled_data
def balance_data(data, target_index):
    # 클래스별 데이터 분포 확인
    class_counts = Counter([row[target_index] for row in data])
    print("Original class distribution:", class_counts)

    # 가장 적은 클래스 데이터 수 찾기
    min_class = min(class_counts, key=class_counts.get)
    min_count = class_counts[min_class]

    # Subsampling: 다수 클래스 데이터를 소수 클래스 데이터 수에 맞춤
    balanced_data = []
    for target_class, count in class_counts.items():
        if count > min_count:
            balanced_data += subsample_data(data, target_index, target_class, min_count)
        else:
            balanced_data += [row for row in data if row[target_index] == target_class]

    # Oversampling: 소수 클래스 데이터를 다수 클래스 데이터 수에 맞춤
    for target_class, count in class_counts.items():
        if count < min_count:
            oversampled_data = oversample_data(data, target_index, target_class, min_count - count)
            balanced_data += oversampled_data

    balanced_class_counts = Counter([row[target_index] for row in balanced_data])
    print("Balanced class distribution:", balanced_class_counts)
    return balanced_data
if __name__ == "__main__":
    # 데이터 읽기
    header, data = read_csv("imbalanced_playtennis.csv")

    # Subsampling과 Oversampling 수행
    balanced_data = balance_data(data, target_index=4)  # target_index는 Play Tennis 컬럼의 인덱스

    # 결과 저장
    write_csv("balanced_playtennis.csv", header, balanced_data)
    print("Balanced data saved to 'balanced_playtennis.csv'")
