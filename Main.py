import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, explode, split, coalesce, lit, regexp_extract, when, last, first
import pandas as pd
import matplotlib.pyplot as plt
from SparkNLPPreprocessorStatic import SparkNLPPreprocessorStatic
from LinkedInDataProcessor import LinkedInDataProcessor
from LinkedInLogisticModel import LinkedInLogisticModel
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main():
    # Initialize Spark session with Spark NLP
    spark = SparkSession.builder \
        .appName("Spark NLP Test") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.2") \
        .getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    processor = LinkedInDataProcessor(
        spark,
        companies_path="/dbfs/linkedin_train_data",
        employees_path="/dbfs/linkedin_people_train_data",
        university_path="/Workspace/Users/dan_amler@campus.technion.ac.il/university_data.csv"
    )
    processed_data = processor.run_pipeline()

    preprocessor = SparkNLPPreprocessorStatic(
        text_cols=["title", "subtitle"],
        numeric_cols=["mean_acceptance_rate", "mean_graduation_rate", "mean_avg_cost_after_aid", "num_past_experience",
                      "num_education", "recommendations_count", "avg_past_months"]
    )

    preprocessor.build_pipeline()
    final_df = preprocessor.fit_transform(processed_data)
    final_df = final_df.filter(
        "title IS NOT NULL AND TRIM(title) != '' "
        "AND subtitle IS NOT NULL AND TRIM(subtitle) != '' "
    )
    model = LinkedInLogisticModel(spark, final_df)
    metrics = model.run_pipeline()


if __name__ == "__main__":
    main()