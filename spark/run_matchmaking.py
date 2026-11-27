"""Τρέχει το matchmaking και τυπώνει τα Top-K. Χρήση:
    python spark/run_matchmaking.py [path/to/profiles.json]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_profiles  # noqa: E402
from matchmaking import compute_partners  # noqa: E402
from spark_config import get_spark  # noqa: E402

if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "mini_profiles.json")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    result = compute_partners(load_profiles(spark, path))
    result.drop("computed_at").show(50, truncate=False)
    spark.stop()
