import os
import csv
import json

def convert_csv_to_json(dataset_dir, output_json):
    splits = ['train', 'val', 'test']
    data = {split: [] for split in splits}

    if os.path.exists(dataset_dir):
        print("all good, proceed")
    else:
        print('get fucked')
        return -1

    for split in splits:
        csv_path = os.path.join(dataset_dir, f"{split}.csv")
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                # Assuming the CSV format is: circuit id, circuit image name, class id
                image_name = row[1]
                class_id = int(row[2])
                label_id = str(class_id)  # Assigning class_id as label_id as per format
                data[split].append([image_name, class_id, label_id])

    # Write data to JSON file
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data has been converted and saved to {output_json}")

# Usage
dataset_dir = 'LITE-benchmark\circuit-diagrams'  # Replace with the directory containing train.csv, val.csv, and test.csv
output_json = os.path.join(dataset_dir, 'split_config.json')
convert_csv_to_json(dataset_dir, output_json)