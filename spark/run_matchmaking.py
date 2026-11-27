"""Τρέχει το matchmaking και τυπώνει τα Top-K. Χρήση (από τη ρίζα του repo):
    python spark/run_matchmaking.py [path/to/profiles.json]

Χωρίς όρισμα χρησιμοποιεί το κοινό δοκιμαστικό αρχείο tests/fixtures/similarity_sample.json.
"""
import sys

from data_loader import SAMPLE_PATH, load_profiles
from matchmaking import compute_partners
from spark_config import create_spark_session

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_PATH
    spark = create_spark_session("SmartHelixMatchmaking")
    try:
        result = compute_partners(load_profiles(spark, path))
        result.drop("computed_at").show(50, truncate=False)
    finally:
        spark.stop()
