import csv
import sys

# Define the CSV file path
csv_file = "progress.csv"

# Function to find the column index for a given column name
def get_column_index(column_name):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        try:
            return header.index(column_name)
        except ValueError:
            print(f"Column '{column_name}' not found in CSV file.")
            sys.exit(1)

# Get the column index for "Average latency"
latency_column_index = get_column_index("Average latency") 
print(latency_column_index)
# Find the minimum value in the "Average latency" column
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    min_latency = float('inf')  # Initialize with infinity
    for row in reader:
        try:
            latency_value = float(row[latency_column_index])
            min_latency = min(min_latency, latency_value)
        except (ValueError, IndexError):
            pass  # Ignore rows where the value cannot be converted to float or index is out of range

if min_latency != float('inf'):
    print(f"Minimum value of Average latency column: {min_latency}")
else:
    print("No valid values found in the column.")

