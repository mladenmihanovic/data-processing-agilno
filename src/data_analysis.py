from pyspark.sql import DataFrame, functions as F
from utils.output_writter import output_results

@output_results(output_filename='distributions/depression_by_demographics.parquet', partition_by=None)
def get_depression_percentage_by_age_group_and_proffession(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("Age Group", "Profession")
        .agg(
            F.mean("Depression").alias("Avg Depression"),
            F.count("Depression").alias("Count"),
        )
        .filter(F.col("Count") > 100)
        .orderBy("Age Group")
    )

@output_results(output_filename='distributions/academic_performance.parquet.parquet', partition_by=None)
def get_CGPA_statistics_by_sleep_category(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("Sleep Category")
        .agg(F.mean("CGPA").alias("Avg CGPA"), F.stddev("CGPA").alias("StdDev CGPA"))
        .orderBy("Sleep Category")
    )

@output_results(output_filename='correlations/correlation_matrix.parquet', partition_by=None)
def get_corelation_matrix(df: DataFrame) -> DataFrame:
    return df

@output_results(output_filename='correlations/depression_correlations.parquet', partition_by=None)
def get_corelation_matrix(df: DataFrame) -> DataFrame:
    return df

@output_results(output_filename='aggregations/city_degree_stats.parquet', partition_by=None)
def get_depression_by_city_and_degree(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("City", "Degree")
        .agg(
            F.round((F.mean("Depression") * 100), 2).alias("Depression Percentage"),
            F.count("City").alias("Count"),
        )
        .orderBy("City", "Degree")
    )

@output_results(output_filename='aggregations/demographic_stress.parquet', partition_by=None)
def get_stress_index_by_age_group_and_gender(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("Age Group", "Gender")
        .agg(
            F.mean("Stress Index").alias("Avg Stress Index"),
            F.count("Age Group").alias("Count"),
        )
        .orderBy("Age Group", "Gender")
    )

@output_results(output_filename='aggregations/sleep_performance.parquet', partition_by=None)
def get_accademic_performance_per_sleep_category(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("Sleep Category")
        .agg(F.mean("CGPA").alias("Avg CGPA"), F.count("Sleep Category").alias("Count"))
        .orderBy("Sleep Category")
    )

@output_results(output_filename='aggregations/high_risk_students.parquet', partition_by=None)
def get_high_risk_students(df: DataFrame) -> DataFrame:
    
    high_stress_threshold = 4
    low_sleep_threshold = 6
    low_job_satisfaction_threshold = 2
    high_academin_pressure_threshold = 4
    high_financial_stress_threshold = 4

    return df.filter(
        (F.col('Stress Index') > high_stress_threshold) &
        (F.col('Sleep Duration') < low_sleep_threshold) &
        (F.col('Job Satisfaction') < low_job_satisfaction_threshold) & 
        (F.col('Academic Pressure') > high_academin_pressure_threshold) &
        (F.col('Financial Stress') > high_financial_stress_threshold)
    )

def output_data_analysis(df: DataFrame) -> None:
    get_depression_percentage_by_age_group_and_proffession(df)
    get_CGPA_statistics_by_sleep_category(df)
    get_corelation_matrix(df)
    get_corelation_matrix(df)
    get_depression_by_city_and_degree(df)
    get_stress_index_by_age_group_and_gender(df)
    get_accademic_performance_per_sleep_category(df)
    get_high_risk_students(df)