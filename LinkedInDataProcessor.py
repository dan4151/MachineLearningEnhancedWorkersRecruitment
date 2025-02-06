from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, posexplode, expr, regexp_extract, when, explode, avg, median, size
)

class LinkedInDataProcessor:
    def __init__(self, spark: SparkSession, companies_path: str, employees_path: str, university_path: str):
        """
        Initializes the LinkedIn Data Processor.
        :param spark: SparkSession object
        :param companies_path: Path to companies parquet data
        :param employees_path: Path to employees parquet data
        :param university_path: Path to university CSV data
        """
        self.spark = spark
        self.companies_path = companies_path
        self.employees_path = employees_path
        self.university_path = university_path
        self.companies = None
        self.employees = None
        self.employees_in_jobs = None
        self.education_data = None
        self.average_past_months = None
        self.full_data = None

    def load_data(self):
        """Loads data from parquet and CSV files."""
        self.companies = self.spark.read.parquet(self.companies_path)
        self.employees = self.spark.read.parquet(self.employees_path)
        self.universities = self.spark.read.csv(f"file://{self.university_path}", header=True, inferSchema=True)

    def preprocess_employees(self):
        """Filters employees with valid experience and education."""
        self.employees = self.employees \
            .where('experience is not null') \
            .where('size(experience) > 0') \
            .where('education is not null') \
            .where('size(education) > 0')

    def extract_experience(self):
        """Extracts job-related information from experience data."""
        self.employees_in_jobs = self.employees \
            .select("*", posexplode(col("experience")).alias("pos", "current_comp")) \
            .withColumn("past_experience", expr("slice(experience, 1, pos)")) \
            .drop("experience")

        self.employees_in_jobs = self.employees_in_jobs.withColumn("duration", col("current_comp.duration"))
        self.employees_in_jobs = self.employees_in_jobs.withColumn("years", regexp_extract(col("duration"), r"(\d+)\s+year", 1).cast("int"))
        self.employees_in_jobs = self.employees_in_jobs.withColumn("months", regexp_extract(col("duration"), r"(\d+)\s+month", 1).cast("int"))
        self.employees_in_jobs = self.employees_in_jobs.fillna({"years": 0, "months": 0})
        self.employees_in_jobs = self.employees_in_jobs.withColumn(
            "moreThanYear",
            when(col("years") >= 1, 1).otherwise(0)
        )

    def extract_education_data(self):
        """Extracts university-related data for employees."""
        exploded_df = self.employees.withColumn("University", explode(col("education.title"))) \
            .select("id", "University") \
            .join(self.universities, on="University", how="left")

        self.education_data = exploded_df.groupBy("id").agg(
            avg("Acceptance Rate").alias("mean_acceptance_rate"),
            avg("Graduation Rate").alias("mean_graduation_rate"),
            avg("Avg Cost After Aid").alias("mean_avg_cost_after_aid")
        ).select("id", "mean_acceptance_rate", "mean_graduation_rate", "mean_avg_cost_after_aid")

    def extract_past_experience(self):
        """Processes past job experiences and computes average duration."""
        exploded_past = self.employees_in_jobs.withColumn("past_duration", explode(col("past_experience.duration_short")))
        exploded_past = exploded_past.withColumn("past_job_years", regexp_extract(col("past_duration"), r"(\d+)\s+year", 1).cast("int"))
        exploded_past = exploded_past.withColumn("past_job_months", regexp_extract(col("past_duration"), r"(\d+)\s+month", 1).cast("int"))
        exploded_past = exploded_past.fillna({"past_job_years": 0, "past_job_months": 0})
        exploded_past = exploded_past.withColumn("total_past_months", col("past_job_years") * 12 + col("past_job_months"))

        self.average_past_months = exploded_past.groupBy("id", "past_experience") \
            .agg(avg("total_past_months").alias("avg_past_months")) \
            .select("id", "past_experience", "avg_past_months")

    def prepare_final_data(self):
        """Combines all processed data into a single DataFrame."""
        self.full_data = self.employees_in_jobs \
            .join(self.education_data, on="id", how="left") \
            .join(self.average_past_months, on=["id", "past_experience"], how="left") \
            .select(
                "moreThanYear",
                "mean_acceptance_rate",
                "mean_graduation_rate",
                "mean_avg_cost_after_aid",
                "current_comp.title",
                "current_comp.subtitle",
                "avg_past_months",
                when(col("recommendations_count").isNotNull(), col("recommendations_count"))
                .otherwise(0).alias("recommendations_count"),
                size("education").alias("num_education"),
                size("past_experience").alias("num_past_experience")
            )

    def fill_missing_values(self):
        """Fills missing values in numeric columns using median values."""
        numeric_cols = [
            "mean_acceptance_rate", "mean_graduation_rate", "mean_avg_cost_after_aid",
            "num_past_experience", "num_education", "avg_past_months"
        ]

        median_values = self.full_data.select(
            *[median(col(c)).alias(f"median_{c}") for c in numeric_cols]
        ).collect()[0]

        fill_dict = {c: median_values[f"median_{c}"] for c in numeric_cols}
        self.full_data = self.full_data.na.fill(fill_dict)

        for c in numeric_cols:
            self.full_data = self.full_data.withColumn(
                c,
                when(col(c).rlike("^[0-9]+(\\.[0-9]*)?$"), col(c))  # Keep numeric values
                .otherwise(None)  # Replace non-numeric with null
                .cast("float")  # Convert to float
            )

    def run_pipeline(self):
        """Runs the entire data processing pipeline."""
        self.load_data()
        self.preprocess_employees()
        self.extract_experience()
        self.extract_education_data()
        self.extract_past_experience()
        self.prepare_final_data()
        self.fill_missing_values()
        return self.full_data

