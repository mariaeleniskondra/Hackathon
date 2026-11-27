from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

from spark_config import create_spark_session
from data_loader import load_profiles


def preprocess_profiles(profiles):
    """
    Prepares business profiles for similarity analysis.
    """

    # 1. Remove profiles with missing IDs
    profiles = profiles.filter(F.col("id").isNotNull())

    # 2. Remove duplicate profile IDs
    profiles = profiles.dropDuplicates(["id"])

    # 3. Replace missing text values
    profiles = profiles.fillna({
        "description": "",
        "category": ""
    })

    # 4. Replace missing arrays with empty arrays
    array_columns = [
        "skills",
        "services",
        "products",
        "needs"
    ]

    for column in array_columns:
        profiles = profiles.withColumn(
            column,
            F.coalesce(
                F.col(column),
                F.array().cast("array<string>")
            )
        )

    # 5. Combine profile characteristics
    profiles = profiles.withColumn(
        "combined_text",
        F.concat_ws(
            " ",
            F.col("description"),
            F.col("category"),
            F.concat_ws(" ", F.col("skills")),
            F.concat_ws(" ", F.col("services")),
            F.concat_ws(" ", F.col("products"))
        )
    )

    # 6. Normalize text
    profiles = profiles.withColumn(
        "clean_text",
        F.trim(
            F.regexp_replace(
                F.lower(F.col("combined_text")),
                r"\s+",
                " "
            )
        )
    )

    # 7. Tokenization
    tokenizer = RegexTokenizer(
        inputCol="clean_text",
        outputCol="tokens",
        pattern=r"\s+",
        minTokenLength=1
    )

    profiles = tokenizer.transform(profiles)

    # 8. Remove English stop words
    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )

    profiles = remover.transform(profiles)

    return profiles


if __name__ == "__main__":

    spark = create_spark_session(
        "SmartHelixPreprocessing"
    )

    try:
        profiles = load_profiles(spark)

        processed = preprocess_profiles(profiles)

        print("Preprocessing completed successfully!")

        print("Number of profiles:", processed.count())

        print("\nCombined text:")

        processed.select(
            "id",
            "name",
            "combined_text"
        ).show(truncate=False)

        print("\nClean text:")

        processed.select(
            "id",
            "name",
            "clean_text"
        ).show(truncate=False)

        print("\nTokens:")

        processed.select(
            "id",
            "name",
            "tokens"
        ).show(truncate=False)

        print("\nFiltered tokens:")

        processed.select(
            "id",
            "name",
            "filtered_tokens"
        ).show(truncate=False)

    finally:
        spark.stop()

