from pathlib import Path
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
    ArrayType
)

from spark_config import create_spark_session


# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Temporary dataset for similarity testing
SAMPLE_PATH = (
    ROOT_DIR / "tests" / "fixtures" / "similarity_sample.json"
)


# Common profile schema
PROFILE_SCHEMA = StructType([
    StructField("id", LongType(), True),
    StructField("external_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("type", StringType(), True),
    StructField("description", StringType(), True),
    StructField("category", StringType(), True),
    StructField("skills", ArrayType(StringType()), True),
    StructField("services", ArrayType(StringType()), True),
    StructField("products", ArrayType(StringType()), True),
    StructField("needs", ArrayType(StringType()), True),
    StructField("location", StringType(), True),
    StructField("source_id", LongType(), True),
    StructField("source", StringType(), True),
    StructField("source_url", StringType(), True),
    StructField("profile_url", StringType(), True),
    StructField("updated_at", StringType(), True)
])


def load_profiles(spark, file_path=SAMPLE_PATH):
    """
    Loads business profiles from a JSON file
    into a Spark DataFrame.
    """

    file_path = Path(file_path).resolve()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    profiles = (
        spark.read
        .schema(PROFILE_SCHEMA)
        .option("multiLine", True)
        .option("mode", "FAILFAST")
        .json(file_path.as_uri())
    )

    return profiles


if __name__ == "__main__":

    spark = create_spark_session("SmartHelixDataLoader")

    try:
        profiles = load_profiles(spark)

        print("Dataset loaded successfully!")
        print("Number of profiles:", profiles.count())

        print("\nDataset schema:")
        profiles.printSchema()

        print("\nBusiness profiles:")
        profiles.select(
            "id",
            "name",
            "category",
            "skills"
        ).show(truncate=False)

    finally:
        spark.stop()
