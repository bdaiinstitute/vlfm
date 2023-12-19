# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

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
            # Ignore empty files
            if os.path.getsize(os.path.join(directory, filename)) == 0:
                continue
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
        table.add_row([cause.replace("did_not_fail", "succeeded!"), count, f"{percentage:.2f}%"])

    print(table)


def calculate_avg_performance(stats: List[Dict[str, Any]]) -> None:
    """
    Calculate the average performance of the agent across all episodes.

    Args:
        stats (List[Dict[str, Any]]): A list of stats for each episode.
    """
    success, spl, soft_spl = [[episode.get(k, -1) for episode in stats] for k in ["success", "spl", "soft_spl"]]

    # Create a table with headers
    table = PrettyTable(["Metric", "Average"])

    # Add each row to the table
    table.add_row(["Success", f"{sum(success) / len(success) * 100:.2f}%"])
    table.add_row(["SPL", f"{sum(spl) / len(spl) * 100:.2f}%"])
    table.add_row(["Soft SPL", f"{sum(soft_spl) / len(soft_spl) * 100:.2f}%"])

    print(table)


def calculate_avg_fail_per_category(stats: List[Dict[str, Any]]) -> None:
    """
    For each possible "target_object", calculate the average failure rate.

    Args:
        stats (List[Dict[str, Any]]): A list of stats for each episode.
    """
    # Create a dictionary to store the fail count and total count for each category
    category_stats = {}

    for episode in stats:
        category = episode["target_object"]
        success = int(episode["success"]) == 1

        if category not in category_stats:
            category_stats[category] = {"fail_count": 0, "total_count": 0}

        category_stats[category]["total_count"] += 1
        if not success:
            category_stats[category]["fail_count"] += 1

    # Create a table with headers
    table = PrettyTable(["Category", "Average Failure Rate"])

    # Add each row to the table
    for category, c_stats in sorted(
        category_stats.items(),
        key=lambda x: x[1]["fail_count"],
        reverse=True,
    ):
        avg_failure_rate = (c_stats["fail_count"] / c_stats["total_count"]) * 100
        table.add_row(
            [
                category,
                f"{avg_failure_rate:.2f}% ({c_stats['fail_count']}/{c_stats['total_count']})",
            ]
        )

    print(table)


def calculate_avg_fail_rate_per_category(stats: List[Dict[str, Any]], failure_cause: str) -> None:
    """
    For each possible "target_object", count the number of times the agent failed due to
    the given failure cause. Then, sum the counts across all categories and use it to
    divide the per category failure count to get the average failure rate for each
    category.

    Args:
        stats (List[Dict[str, Any]]): A list of stats for each episode.
    """
    category_to_fail_count = {}
    total_fail_count = 0
    for episode in stats:
        if episode["failure_cause"] != failure_cause:
            continue
        total_fail_count += 1
        category = episode["target_object"]
        if category not in category_to_fail_count:
            category_to_fail_count[category] = 0
        category_to_fail_count[category] += 1

    # Create a table with headers
    table = PrettyTable(["Category", f"% Occurrence for {failure_cause}"])

    # Sort the categories by their failure count in descending order
    sorted_categories = sorted(category_to_fail_count.items(), key=lambda x: x[1], reverse=True)

    # Add each row to the table
    for category, count in sorted_categories:
        percentage = (count / total_fail_count) * 100
        table.add_row([category, f"{percentage:.2f}% ({count})"])

    print(table)


def main() -> None:
    """
    Main function to parse command line arguments and process the directory.
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("directory", type=str, help="Directory to process")
    parser.add_argument("--compact", "-c", action="store_true", help="Compact output")
    args = parser.parse_args()

    episode_stats = read_json_files(args.directory)
    print(f"\nTotal episodes: {len(episode_stats)}\n")

    print()
    calculate_avg_performance(episode_stats)

    if "failure_cause" in episode_stats[0]:
        failure_causes = [episode["failure_cause"] for episode in episode_stats]
        calculate_frequencies(failure_causes)

    if args.compact:
        return

    print()
    calculate_avg_fail_per_category(episode_stats)

    print()
    print("Conditioned on failure cause: false_positive")
    calculate_avg_fail_rate_per_category(episode_stats, "false_positive")

    print()
    print("Conditioned on failure cause: false_negative")
    calculate_avg_fail_rate_per_category(episode_stats, "false_negative")


if __name__ == "__main__":
    main()
