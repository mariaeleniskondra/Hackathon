from pyspark.sql import SparkSession


def create_spark_session(app_name="SmartHelix"):
    """
    Creates and returns a Spark Session
    for the Smart Helix project.
    """

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


if __name__ == "__main__":
    spark = create_spark_session()

    print("Spark initialized successfully!")
    print("Spark version:", spark.version)

    spark.range(5).show()

    spark.stop()
