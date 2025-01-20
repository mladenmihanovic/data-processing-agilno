from pyspark.sql import DataFrame
import os

OUTPUTS_DIR = 'outputs'

def output_results(output_filename: str, partition_by: list):
    """
    A decorator to write the results of a function to a Parquet file.

    Args:
        output_filename (str): The name of the output Parquet file.
        partition_by (list): A list of columns to partition the Parquet file by.

    Returns:
        function: A decorator that wraps the original function, writes its result to a Parquet file, and returns the result.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            create_parquet_file(result, output_filename, partition_by)
            return result

        return wrapper

    return decorator


def create_parquet_file(dataFrame: DataFrame, output_file_name: str, partition_by: list) -> None:
    """
    Creates a Parquet file from the given DataFrame.

    Parameters:
    dataFrame (DataFrame): The DataFrame to be written to a Parquet file.
    output_file_name (str): The name of the output Parquet file.
    partition_by (list): A list of columns to partition the data by.

    Returns:
    None
    """
    try:
        processed_data_path = os.path.join(OUTPUTS_DIR, output_file_name)
        writter = dataFrame.write.mode("overwrite")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writter.parquet(processed_data_path)
    except Exception as e:
        print(f"Error writing data to Parquet file: {output_file_name}!\n{str(e)}")
