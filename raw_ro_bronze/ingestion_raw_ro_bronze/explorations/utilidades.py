# Databricks notebook source
# MAGIC %md
# MAGIC Criação de volumes:

# COMMAND ----------

spark.sql("CREATE VOLUME IF NOT EXISTS sandbox.vendas.dspy_databricks_learning;")

# COMMAND ----------

# MAGIC %md
# MAGIC Dropar tabela:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE sandbox.vendas.b_2_w_reviews_bronze;
