import csv
import sys

def read_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')

        data = [row for row in reader]
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python .py .csv")
        sys.exit(1)
    
    data = read_csv(sys.argv[1])
    print(type(data[0]))