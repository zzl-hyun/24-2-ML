# Create multiple test cases with diverse scenarios
import pandas as pd
test_cases = [
    {
        "Outlook": ["Sunny", "Overcast", "Rain", "Sunny", "Rain"],
        "Temperature": ["Hot", "Cool", "Mild", "Hot", "Cool"],
        "Humidity": ["High", "Normal", "High", "High", "Normal"],
        "Wind": ["Weak", "Strong", "Weak", "Strong", "Weak"]
    },
    {
        "Outlook": ["Sunny", "Sunny", "Rain", "Overcast", "Rain"],
        "Temperature": ["Hot", "Mild", "Cool", "Mild", "Cool"],
        "Humidity": ["High", "High", "Normal", "High", "Normal"],
        "Wind": ["Strong", "Weak", "Strong", "Weak", "Weak"]
    },
    {
        "Outlook": ["Rain", "Overcast", "Sunny", "Overcast", "Rain"],
        "Temperature": ["Mild", "Cool", "Hot", "Mild", "Cool"],
        "Humidity": ["Normal", "Normal", "High", "High", "Normal"],
        "Wind": ["Weak", "Weak", "Strong", "Strong", "Weak"]
    }
]

# Save each test case as a CSV file
file_paths = []
for i, test_case in enumerate(test_cases, start=1):
    df = pd.DataFrame(test_case)
    file_path = f'./testcase_{i}.csv'
    df.to_csv(file_path, index=False)
    file_paths.append(file_path)

file_paths
