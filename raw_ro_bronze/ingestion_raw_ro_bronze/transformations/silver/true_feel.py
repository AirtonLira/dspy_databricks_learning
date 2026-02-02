from pyspark import pipelines as dp
from pyspark.sql.functions import *


# ========================================
# SILVER LAYER - Atribui nome do produto e sentimento real com IA
# ========================================
# @dp.materialized_view(
#     name="b_2_w_true_sentiment",
#     comment="Valida através de uma AI o verdadeiro sentimento e qual produto foi mencionado"
# )
# def positive_or_negative_raw():
#     df = spark.read.table("workspace.b2w.b_2_w_reviews_with_sentiment")

#     df_clean_and_sentiment = df.select("reviewer_id", "overall_rating", "review_text", "sentiment")

#     return df