import argparse
import json
import os
from collections import Counter


def read_json_files(directory):
    failure_causes = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                data = json.load(f)
                if "failure_cause" in data:
                    failure_causes.append(data["failure_cause"])
    return failure_causes


def calculate_frequencies(failure_causes):
    counter = Counter(failure_causes)
    total = sum(counter.values())
    for cause, count in counter.items():
        percentage = (count / total) * 100
        print(
            f"Failure cause: {cause}, Frequency: {count}, Percentage: {percentage:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("directory", type=str, help="Directory to process")
    args = parser.parse_args()

    failure_causes = read_json_files(args.directory)
    calculate_frequencies(failure_causes)


if __name__ == "__main__":
    main()
