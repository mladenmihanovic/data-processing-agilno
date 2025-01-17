from pyspark.sql import DataFrame
import os

def output_to_parquet_file(dataFrame: DataFrame, output_file_name) -> None:
    processed_data_path = os.path.join('output', output_file_name)
    dataFrame.write.parquet(processed_data_path)