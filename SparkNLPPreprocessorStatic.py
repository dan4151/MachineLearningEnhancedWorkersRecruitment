import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, explode, split, coalesce, lit, regexp_extract, when, last, first
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import VectorUDT, Vectors
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, WordEmbeddingsModel, LemmatizerModel


class SparkNLPPreprocessorStatic:
    """
    A single class that:
    1. Builds a Spark NLP pipeline to create GloVe embeddings for each specified text column.
    2. Pools (averages) token embeddings to produce a single vector per row/column.
    3. Converts arrays to Spark ML dense vectors.
    4. Combines text vectors (and optional numeric columns) into a `features` column.
    """

    def __init__(
            self,
            embedding_model: str = "glove_100d",
            text_cols: list = None,
            numeric_cols: list = None,
            output_col: str = "features"
    ):
        """
        :param embedding_model: Name of the pretrained GloVe model, e.g., "glove_100d", "glove_300d", etc.
        :param text_cols: List of string columns to embed. Defaults to None.
        :param numeric_cols: List of numeric columns to include in the final feature vector. Defaults to None.
        :param output_col: Name of the final assembled features column. Defaults to "features".
        """
        self.embedding_model_name = embedding_model
        self.text_cols = text_cols or []
        self.numeric_cols = numeric_cols or []
        self.output_col = output_col

        self.pipeline = None
        self._avg_emb_udf = None
        self._to_vec_udf = None

        # Prepare the UDFs needed for manual pooling
        self._create_udfs()

    def _create_pipeline_stages(self):
        """
        Creates Spark NLP pipeline stages for each text column:
          - DocumentAssembler
          - Tokenizer
          - GloVe WordEmbeddingsModel
        """
        stages = []
        for col_name in self.text_cols:
            assembler = DocumentAssembler() \
                .setInputCol(col_name) \
                .setOutputCol(f"{col_name}_document")

            tokenizer = Tokenizer() \
                .setInputCols([f"{col_name}_document"]) \
                .setOutputCol(f"{col_name}_tokens")

            # Add Lemmatizer
            lemmatizer = LemmatizerModel.pretrained("lemma_antbnc", "en") \
                .setInputCols([f"{col_name}_tokens"]) \
                .setOutputCol(f"{col_name}_lemmas")

            embeddings = WordEmbeddingsModel.pretrained(self.embedding_model_name, "en") \
                .setInputCols([f"{col_name}_document", f"{col_name}_lemmas"]) \
                .setOutputCol(f"{col_name}_embeddings")

            stages += [assembler, tokenizer, lemmatizer, embeddings]
        return stages

    def build_pipeline(self):
        """
        Builds the Spark NLP pipeline (DocumentAssembler, Tokenizer, WordEmbeddingsModel)
        for each text column.
        """
        stages = self._create_pipeline_stages()
        self.pipeline = Pipeline(stages=stages)

    def _create_udfs(self):
        """
        Creates the UDFs required for:
          1. Averaging token embeddings (array of annotation structs -> array of floats)
          2. Converting array of floats -> Spark ML dense vector
        """

        # 1) UDF to average token embeddings
        def average_embeddings(annotations):
            """
            annotations: list of annotation structs (each has .embeddings = array[float]).
            Returns a single average embedding array for that row/column.
            """
            if not annotations:
                return []
            vectors = [anno.embeddings for anno in annotations if anno.embeddings is not None]
            if not vectors:
                return []
            arr = np.array(vectors)
            avg_vec = arr.mean(axis=0)
            return avg_vec.tolist()

        self._avg_emb_udf = udf(average_embeddings, ArrayType(FloatType()))

        # 2) UDF to convert float arrays -> DenseVector
        def to_dense_vector(float_array):
            return Vectors.dense(float_array) if float_array else Vectors.dense([])

        self._to_vec_udf = udf(to_dense_vector, VectorUDT())

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        1) Fits the Spark NLP pipeline on the DataFrame (generates token-level embeddings).
        2) Manually averages those embeddings for each text column.
        3) Converts arrays into Spark ML vectors.
        4) Uses VectorAssembler to create a final features column: <self.output_col>.

        Returns a new DataFrame with a "features" column + original columns (including `moreThanYear`).
        """
        if not self.pipeline:
            raise ValueError("Pipeline has not been built. Call build_pipeline() first.")

        # 1) Run the pipeline to produce *_embeddings columns
        model = self.pipeline.fit(df)
        embedded_df = model.transform(df)

        # 2) For each text column, average the token-level embeddings
        pooled_df = embedded_df
        for col_name in self.text_cols:
            pooled_df = pooled_df.withColumn(
                f"{col_name}_vec_array",
                self._avg_emb_udf(col(f"{col_name}_embeddings"))
            )

        # 3) Convert each array of floats into Spark ML DenseVector
        for col_name in self.text_cols:
            pooled_df = pooled_df.withColumn(
                f"{col_name}_vec",
                self._to_vec_udf(col(f"{col_name}_vec_array"))
            )

        # 4) Assemble text vectors + numeric columns into a single features vector
        assembler_input = [f"{c}_vec" for c in self.text_cols] + self.numeric_cols
        assembler = VectorAssembler(
            inputCols=assembler_input,
            outputCol=self.output_col
        )
        final_df = assembler.transform(pooled_df)

        return final_df
