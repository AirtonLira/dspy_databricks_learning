from pyspark import pipelines as dp
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType

@dp.materialized_view(
    name="bronze_b_2_w_reviews",
    comment="Base de reviews pronta com avaliação de sentimento pela LLM"
)
def bronze_b2w_reviews():
    df = spark.read.table("raw_b_2_w_reviews")
    df_clean = df.filter(col("overall_rating") > 1)
    df_clean = df_clean.filter(col("review_text").isNotNull())
    df_clean = df_clean.withColumnRenamed("overall_rating", "sentiment")
    return df_clean
