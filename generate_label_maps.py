from json import load, dump
import argparse
import os
import csv
from pathlib import Path

def create_label_map_from_split(input_file, output):

    if os.path.isdir(output):
        output_file = os.path.join(output, "label_map.json")
    else:
        output_file = output


    with open(input_file, 'r') as myFile:
        data = load(myFile)

    label_map = {}

    for split in ['train', 'val', 'test']:
        for sample in data[split]:
            id = sample[1]
            name = sample[2]

            if id not in label_map:
                label_map[id] = name

    with open(output_file, 'w') as myFile:
        dump(label_map, myFile, indent=4)
        myFile.write("\n")

def create_label_map_from_csv(input_file, output):
    if os.path.isdir(output):
        output_file = os.path.join(output, "label_map.json")
    else:
        output_file = output  
    
    label_map = {}
        
    with open(input_file, 'r') as myFile:
        csv_file = csv.reader(myFile)
        header = next(csv_file)
        for row in csv_file:
            id = row[0]
            label = row[1]

            if id not in label_map:
                label_map[id] = label

    with open(output_file, 'w') as myFile:
        dump(label_map, myFile, indent=4)
        myFile.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="path to split JSON")
    parser.add_argument("output", type=str, help="path to directory where you want to store the mapping")
    args = parser.parse_args()

    if Path(args.input_file).suffix in [".csv"]:
        create_label_map_from_csv(args.input_file, args.output)
    else:
        create_label_map_from_split(args.input_file, args.output)