# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLPath")

    # Oracle stop
    parser.add_argument("--enable_oracle_stop", action="store_true", default=False)

    # Logging etc
    parser.add_argument(
        "--analysis_save_location", type=str, default="failure_analysis/"
    )
    parser.add_argument(
        "--disable_log_success_if_oracle_stop",
        dest="enable_log_success_if_oracle_stop",
        action="store_false",
        default=True,
    )

    # Replanning
    parser.add_argument("--enable_replan_at_steps", action="store_true", default=False)
    parser.add_argument("--replan_interval", type=int, default=30)

    parser.add_argument(
        "--disable_replan_when_stuck",
        dest="enable_replan_when_stuck",
        action="store_false",
        default=True,
    )

    parser.add_argument("--force_dont_stop_until", type=int, default=33)

    parser.add_argument(
        "--force_dont_stop_after_stuck", action="store_true", default=False
    )

    # Calculating similarity
    parser.add_argument(
        "-path_similarity_method",
        type=str,
        default="weighted average embeddings",
        choices=["average sim", "average embeddings", "weighted average embeddings"],
    )

    parser.add_argument("--path_thresh_switch", type=float, default=0.1)
    parser.add_argument("--path_thresh_stop", type=float, default=0.5)

    parser.add_argument(
        "--disable_peak_threshold",
        dest="enable_peak_threshold",
        action="store_false",
        default=True,
    )
    parser.add_argument("--path_thresh_peak", type=float, default=0.9)

    parser.add_argument("--path_prev_val_weight", type=float, default=1.0)

    # For multiresolution
    parser.add_argument("--path_weight_path", default=1.0)
    parser.add_argument("--path_weight_sentence", default=0.6)
    parser.add_argument("--path_weight_parts", default=0.3)
    parser.add_argument("--path_weight_words", default=0.6)

    parser.add_argument("--path_thresh_peak_parts_val", default=0.9)
    parser.add_argument("--path_thresh_peak_parts_switch", default=0.7)

    # Stairs
    parser.add_argument(
        "--disable_stairs", dest="enable_stairs", action="store_false", default=True
    )

    # Map
    parser.add_argument(
        "--vl_feature_type",
        type=str,
        default="BLIP2",
        choices=["BLIP2", "CLIP", "Lseg"],
    )

    parser.add_argument("--map_size", type=int, default=1000)

    args = parser.parse_args()
    return args
