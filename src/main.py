from pyspark.sql import SparkSession
from data_processor import clean_and_prepare_data
from feature_engineering import transform_and_enrich_data
from data_analysis import output_data_analysis

def load_data(spark: SparkSession, file_path: str):
    """
    Load data from a CSV file into a Spark DataFrame.

    Args:
        spark (SparkSession): The SparkSession object.
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: A Spark DataFrame containing the loaded data.
    """
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

def main() -> None:
    """
    Main function to execute the data processing pipeline.

    This function performs the following steps:
    1. Initializes a Spark session.
    2. Loads the data from a CSV file.
    3. Cleans and prepares the data.
    4. Transforms and enriches the data.
    5. Outputs the data analysis results.
    6. Closes the Spark session.

    Returns:
        None
    """
    spark = SparkSession.builder.getOrCreate()
    df = load_data(spark, '../data/Student Depression Dataset.csv')
    df = clean_and_prepare_data(df)
    transformed_df = transform_and_enrich_data(df)
    output_data_analysis(transformed_df)
    spark.stop()


if __name__ == '__main__':
    main()