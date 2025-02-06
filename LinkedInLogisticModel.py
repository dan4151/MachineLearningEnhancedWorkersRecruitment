from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class LinkedInLogisticModel:
    def __init__(self, spark: SparkSession, df):
        """
        Initializes the LinkedIn Logistic Regression Model.
        :param spark: SparkSession object
        :param df: Preprocessed Spark DataFrame containing features and labels
        """
        self.spark = spark
        self.df = df
        self.weights = None
        self.model = None
        self.train_df = None
        self.test_df = None
        self.predictions = None
        self.evaluator = BinaryClassificationEvaluator(labelCol="moreThanYear", rawPredictionCol="rawPrediction")

    def compute_class_weights(self):
        """Computes class weights to handle imbalanced classes."""
        total_count = self.df.count()
        class_counts = self.df.groupBy("moreThanYear").count().rdd.collectAsMap()
        self.weights = {k: total_count / (2.0 * v) for k, v in class_counts.items()}
        print(f"Class weights: {self.weights}")

        # Define UDF to map weights
        def calculate_weight(more_than_year):
            return float(self.weights[more_than_year])

        weight_udf = udf(calculate_weight, FloatType())

        # Add weight column
        self.df = self.df.withColumn("classWeight", weight_udf(col("moreThanYear")))

    def split_data(self, train_ratio=0.8):
        """Splits the dataset into training and test sets."""
        self.train_df, self.test_df = self.df.randomSplit([train_ratio, 1 - train_ratio], seed=42)

    def train_model(self):
        """Trains a logistic regression model."""
        lr = LogisticRegression(featuresCol="features", labelCol="moreThanYear", weightCol="classWeight")
        self.model = lr.fit(self.train_df)

        print("Model trained successfully.")
        print("Coefficients:", self.model.coefficients)
        print("Intercept:", self.model.intercept)

        feature_names = [
            "mean_acceptance_rate", "mean_graduation_rate", "mean_avg_cost_after_aid",
            "num_past_experience", "num_education", "recommendations_count", "avg_past_months"
        ]

        print("\nFeature Coefficients:")
        for c, val in zip(feature_names, self.model.coefficients[-7:]):
            print(f"{c}: {val}")

    def evaluate_model(self):
        """Evaluates the model using AUC, F1 score, and accuracy."""
        self.predictions = self.model.transform(self.test_df)

        # Compute AUC
        auc = self.evaluator.evaluate(self.predictions)
        print(f"AUC on Test Data: {auc}")

        # Compute confusion matrix counts
        count_11 = self.predictions.filter((col("moreThanYear") == 1) & (col("prediction") == 1)).count()
        count_10 = self.predictions.filter((col("moreThanYear") == 1) & (col("prediction") == 0)).count()
        count_01 = self.predictions.filter((col("moreThanYear") == 0) & (col("prediction") == 1)).count()
        count_00 = self.predictions.filter((col("moreThanYear") == 0) & (col("prediction") == 0)).count()

        # Compute F1 score
        precision = count_11 / (count_11 + count_10) if (count_11 + count_10) > 0 else 0
        recall = count_11 / (count_11 + count_01) if (count_11 + count_01) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Compute accuracy
        accuracy = (count_11 + count_00) / (count_11 + count_10 + count_01 + count_00)

        print(f"F1 Score: {f1}")
        print(f"Accuracy: {accuracy}")

        return {"AUC": auc, "F1 Score": f1, "Accuracy": accuracy}

    def run_pipeline(self):
        """Runs the full ML pipeline: weighting, splitting, training, and evaluation."""
        self.compute_class_weights()
        self.split_data()
        self.train_model()
        return self.evaluate_model()
