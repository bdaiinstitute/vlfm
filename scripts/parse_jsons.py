import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List

from prettytable import PrettyTable


def read_json_files(directory: str) -> List[Dict[str, Any]]:
    """
    Read JSON files from a directory and extract the failure causes.

    Args:
        directory (str): The directory containing the JSON files.

    Returns:
        List[str]: A list of failure causes.
    """
    episode_stats = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                episode_stats.append(json.load(f))
    return episode_stats


def calculate_frequencies(failure_causes: List[str]) -> None:
    """
    Calculate the frequencies of failure causes and print them.

    Args:
        failure_causes (List[str]): A list of failure causes.
    """
    counter = Counter(failure_causes)
    total = sum(counter.values())

    # Create a table with headers
    table = PrettyTable(["Failure Cause", "Frequency", "Percentage"])

    for cause, count in counter.most_common():
        percentage = (count / total) * 100
        # Add each row to the table
        table.add_row(
            [cause.replace("did_not_fail", "succeeded!"), count, f"{percentage:.2f}%"]
        )

    print(table)


def calculate_avg_performance(stats: List[Dict[str, Any]]) -> None:
    """
    Calculate the average performance of the agent across all episodes.

    Args:
        stats (List[Dict[str, Any]]): A list of stats for each episode.
    """
    success, spl, soft_spl = [
        [episode[k] for episode in stats] for k in ["success", "spl", "soft_spl"]
    ]

    # Create a table with headers
    table = PrettyTable(["Metric", "Average"])

    # Add each row to the table
    table.add_row(["Success", f"{sum(success) / len(success) * 100:.2f}%"])
    table.add_row(["SPL", f"{sum(spl) / len(spl) * 100:.2f}%"])
    table.add_row(["Soft SPL", f"{sum(soft_spl) / len(soft_spl) * 100:.2f}%"])

    print(table)


def main() -> None:
    """
    Main function to parse command line arguments and process the directory.
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("directory", type=str, help="Directory to process")
    args = parser.parse_args()

    episode_stats = read_json_files(args.directory)
    print(f"\nTotal episodes: {len(episode_stats)}\n")

    failure_causes = [episode["failure_cause"] for episode in episode_stats]
    calculate_frequencies(failure_causes)

    print()
    calculate_avg_performance(episode_stats)


if __name__ == "__main__":
    main()
