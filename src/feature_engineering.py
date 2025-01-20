from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, udf, when, lit
from pyspark.sql.types import FloatType


def calculate_stress_index(
    academic_pressure: float, work_pressure: float, financial_stress: float
) -> float:
    """
    Calculate the stress index based on academic, work, and financial pressures.

    This function computes a stress index by taking into account the academic pressure,
    work pressure, and financial stress, each weighted differently.

    Args:
        academic_pressure (float): The level of academic pressure.
        work_pressure (float): The level of work pressure.
        financial_stress (float): The level of financial stress.

    Returns:
        float: The calculated stress index.
    """
    return academic_pressure * 0.4 + work_pressure * 0.4 + financial_stress * 0.6


def create_stress_index(df: DataFrame) -> DataFrame:
    """
    Adds a "Stress Index" column to the given DataFrame by calculating the stress index
    based on "Academic Pressure", "Work Pressure", and "Financial Stress" columns.

    Parameters:
    df (DataFrame): Input DataFrame containing the columns "Academic Pressure",
                    "Work Pressure", and "Financial Stress".

    Returns:
    DataFrame: A new DataFrame with an additional "Stress Index" column, rounded to 2 decimal places.
    """
    calculate_stress_index_udf = udf(calculate_stress_index, FloatType())
    return df.withColumn(
        "Stress Index",
        F.round(
            calculate_stress_index_udf(
                col("Academic Pressure"), col("Work Pressure"), col("Financial Stress")
            ),
            2,
        ),
    )


def get_sleep_category(sleep_duration: float) -> str:
    """
    Categorize sleep duration into 'Low', 'Normal', or 'High'.

    Parameters:
        sleep_duration (float): The duration of sleep in hours.

    Returns:
        str: The sleep category based on the duration.
            - "Low" for less than 6 hours
            - "Normal" for 6 to 8 hours
            - "High" for more than 8 hours
    """
    if sleep_duration < 6:
        return "Low"
    elif 6 <= sleep_duration <= 8:
        return "Normal"
    else:
        return "High"


def create_sleep_categories(df: DataFrame) -> DataFrame:
    """
    Adds a new column "Sleep Category" to the DataFrame based on the "Sleep Duration" column.

    Parameters:
    df (DataFrame): The input DataFrame containing a "Sleep Duration" column.

    Returns:
    DataFrame: A new DataFrame with an additional "Sleep Category" column.
    """

    sleep_category_udf = udf(get_sleep_category, FloatType())
    return df.withColumn("Sleep Category", sleep_category_udf(col("Sleep Duration")))


def create_age_groups(df: DataFrame) -> DataFrame:
    """
    Adds an "Age Group" column to the DataFrame based on the "Age" column.

    Parameters:
    df (DataFrame): Input DataFrame containing an "Age" column.

    Returns:
    DataFrame: DataFrame with an additional "Age Group" column.
    """
    return df.withColumn(
        "Age Group",
        when((col("Age") >= 18) & (col("Age") <= 21), "18-21")
        .when((col("Age") >= 22) & (col("Age") <= 25), "22-25")
        .when((col("Age") >= 26) & (col("Age") <= 30), "26-30")
        .otherwise("30+"),
    )


def create_normalized_columns(df: DataFrame) -> DataFrame:
    """
    Normalize specified numeric columns in the given DataFrame.

    This function takes a DataFrame and normalizes the values of specified numeric columns
    by scaling them to a range between 0 and 1. The normalized values are added as new columns
    with the suffix '_normalized'.

    Args:
        df (DataFrame): The input DataFrame containing the data to be normalized.

    Returns:
        DataFrame: A new DataFrame with the normalized columns added.

    Example:
        >>> df = spark.createDataFrame([(1,), (2,), (3,)], ["Stress Index"])
        >>> result_df = create_normalized_columns(df)
        >>> result_df.show()
        +------------+-------------------+
        |Stress Index|Stress Index_normalized|
        +------------+-------------------+
        |           1|                0.00|
        |           2|                0.50|
        |           3|                1.00|
        +------------+-------------------+
    """

    numeric_cols = ["Stress Index"]

    for col_name in numeric_cols:
        min_val = df.agg(F.min(col_name)).collect()[0][0]
        max_val = df.agg(F.max(col_name)).collect()[0][0]

        return df.withColumn(
            f"{col_name}_normalized",
            F.round((F.col(col_name) - min_val) / (max_val - min_val), 2),
        )


def create_dummy_variables(df: DataFrame) -> DataFrame:
    """
    Creates dummy variables for categorical columns in the given DataFrame.

    This function performs one-hot encoding on the 'Gender', 'Age Group', and
    'Sleep Category' columns of the input DataFrame. It pivots these columns
    to create separate columns for each unique value in these categories,
    filling missing values with 0. The resulting DataFrame is then joined
    with the original DataFrame on the 'id' column.

    Args:
        df (DataFrame): The input DataFrame containing the columns 'id',
                        'Gender', 'Age Group', and 'Sleep Category'.

    Returns:
        DataFrame: A new DataFrame with the original columns and additional
                   dummy variable columns for 'Gender', 'Age Group', and
                   'Sleep Category'.
    """

    gender_pivot_df = df.groupBy("id").pivot("Gender").agg(lit(1)).fillna(0)
    age_group_pivot_df = df.groupBy("id").pivot("Age Group").agg(lit(1)).fillna(0)
    sleep_category_pivot_df = (
        df.groupBy("id").pivot("Sleep Category").agg(lit(1)).fillna(0)
    )

    return (
        df.join(gender_pivot_df, on="id", how="inner")
        .join(age_group_pivot_df, on="id", how="inner")
        .join(sleep_category_pivot_df, on="id", how="inner")
    )


def transform_and_enrich_data(df: DataFrame) -> DataFrame:
    """
    Transforms and enriches the input DataFrame by applying a series of feature engineering functions.

    Args:
        df (DataFrame): The input DataFrame to be transformed and enriched.

    Returns:
        DataFrame: The transformed and enriched DataFrame.

    The following transformations are applied in sequence:
        1. create_stress_index: Adds a stress index column to the DataFrame.
        2. create_sleep_categories: Categorizes sleep data and adds corresponding columns.
        3. create_age_groups: Groups age data and adds corresponding columns.
        4. create_normalized_columns: Normalizes specified columns.
        5. create_dummy_variables: Converts categorical variables into dummy/indicator variables.
    """
    transformed_df = df
    transformed_df = create_stress_index(transformed_df)
    transformed_df = create_sleep_categories(transformed_df)
    transformed_df = create_age_groups(transformed_df)
    transformed_df = create_normalized_columns(transformed_df)
    transformed_df = create_dummy_variables(transformed_df)
    return transformed_df
