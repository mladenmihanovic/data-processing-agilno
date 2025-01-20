from pyspark.sql import DataFrame
from pyspark.sql.functions import when, col


def handleMissingValues(df: DataFrame) -> DataFrame:
    """
    Handles missing values in the given DataFrame.

    This function drops rows where the 'Financial Stress' column has missing values.

    Parameters:
    df (DataFrame): The input DataFrame containing the data.

    Returns:
    DataFrame: A DataFrame with rows containing missing values in the 'Financial Stress' column removed.
    """
    # return df.na.fill(value=0,subset=['Financial Stress']).
    return df.na.drop(subset=["Financial Stress"])


def transformSleepDuration(df: DataFrame) -> DataFrame:
    """
    Transforms the 'Sleep Duration' column in the given DataFrame by mapping string values to numerical values.

    Args:
        df (DataFrame): Input DataFrame containing a 'Sleep Duration' column with string values.

    Returns:
        DataFrame: Transformed DataFrame with 'Sleep Duration' column as float values.
    """
    df_transformed = df.filter('`Sleep Duration` != "Others"')
    df_transformed = df_transformed.withColumn(
        "Sleep Duration",
        when(col("Sleep Duration") == "More than 8 hours", 9)
        .when(col("Sleep Duration") == "7-8 hours", 7.5)
        .when(col("Sleep Duration") == "5-6 hours", 5.5)
        .when(col("Sleep Duration") == "Less than 5 hours", 4),
    )

    return df_transformed.withColumn("Sleep Duration", col("Sleep Duration").cast("float"))


def clean_and_prepare_data(df: DataFrame) -> DataFrame:
    """
    Cleans and prepares the given DataFrame by handling missing values and transforming sleep duration.

    Args:
        df (DataFrame): The input DataFrame to be processed.

    Returns:
        DataFrame: The processed DataFrame with handled missing values and transformed sleep duration.
    """
    df_processed = df
    df_processed = handleMissingValues(df_processed)
    df_processed = transformSleepDuration(df_processed)

    return df_processed
