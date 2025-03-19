# Databricks notebook source
# MAGIC %pip install /Volumes/mlcore_dev/mlcore_init_scripts/mlworkspace/MLCORE_INIT/monitor_db_uc/MLCoreSDK_monitor_db_uc-0.4.6-py3-none-any.whl --force-reinstall
# MAGIC %pip install sparkmeasure

# COMMAND ----------

# DBTITLE 1,Imports
from MLCORE_SDK import mlclient
import ast
# from pyspark.sql import functions as F
from datetime import datetime
from delta.tables import *
import time
import pandas as pd
import mlflow
# import pickle
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.window import Window

# COMMAND ----------

from sparkmeasure import StageMetrics
from sparkmeasure import TaskMetrics
taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Model & Table parameters
# MAGIC

# COMMAND ----------

import yaml
import json
from MLCORE_SDK import mlclient
# from pyspark.sql import functions as F
# import pickle

try:
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = json.loads(solution_config)
    print("Loaded Solution Config from job params")
except Exception as e:
    print(e)
    with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

tracking_env = solution_config["general_configs"]["tracking_env"]
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]
 
tracking_url = solution_config["general_configs"].get("tracking_url", None)
tracking_url = f"https://{tracking_url}" if tracking_url else None

use_latest = True

# JOB SPECIFIC PARAMETERS FOR INFERENCE
input_table_configs = solution_config["inference"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["inference"]["datalake_configs"]['output_tables']
model_configs = solution_config["inference"]["model_configs"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
is_scheduled = solution_config["inference"]["is_scheduled"]
batch_size = int(solution_config["inference"].get("batch_size",500))
cron_job_schedule = solution_config["inference"].get("cron_job_schedule","0 */10 * ? * *")

# COMMAND ----------

# GENERAL PARAMETER
def get_name_space(table_config):
    data_objects = {}
    for table_name, config in table_config.items() : 
        catalog_name = config.get("catalog_name", None)
        schema = config.get("schema", None)
        table = config.get("table", None)

        if catalog_name and catalog_name.lower() != "none":
            table_path = f"{catalog_name}.{schema}.{table}"
        else :
            table_path = f"{schema}.{table}"

        data_objects[table_name] = table_path
    
    return data_objects

# COMMAND ----------

input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

import calendar
from datetime import timedelta
from pyspark.sql.functions import min, max

# COMMAND ----------

# Table Exists or Not
def table_already_created(catalog_name, db_name, table_name):
    db_name = f"{catalog_name}.{db_name}" if catalog_name else db_name
    table_exists = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]
    return any(table_exists)

# Create SQL Query
def get_task_logger(catalog_name, db_name, table_name):
    if table_already_created(catalog_name, db_name, table_name): 
        result = spark.sql(f"SELECT * FROM {catalog_name}.{db_name}.{table_name} ORDER BY timestamp desc LIMIT 1").collect()
        if result:
            task_logger = result[0].asDict()
            return task_logger["start_marker"], task_logger["end_marker"]
    return 0, 0

def get_the_batch_data(catalog_name, db_name, source_data_path, task_logger_table_name):
    start_marker, end_marker = get_task_logger(catalog_name, db_name, task_logger_table_name)
    query_date = f"SELECT * FROM {source_data_path}"
    if start_marker and end_marker:
        query_date += f" WHERE {generate_filter_condition(start_marker, end_marker)}"
    filtered_df = spark.sql(query_date)
    first_record = filtered_df.first()
    first_date = first_record['date']

    # Get the last day of the month dynamically
    last_day = calendar.monthrange(first_date.year, first_date.month)[1]
    new_date = first_date.replace(day=last_day)  # Set new_date to the last date of that month
    # new_date = first_date + timedelta(days=30)
    
    filtered_df = filtered_df.filter((F.col('date') >= first_date) & (F.col('date') <= new_date))
    min_id = filtered_df.agg(min('id')).collect()[0][0]
    max_id = filtered_df.agg(max('id')).collect()[0][0]
    print("Minimum ID:", min_id)
    print("Maximum ID:", max_id)
    batch_size=max_id-min_id+1
    print(f"Batch Size : {batch_size}")

    query = f"SELECT * FROM {source_data_path}"
    if start_marker and end_marker:
        query += f" WHERE {generate_filter_condition(start_marker, end_marker)}"
    query += " ORDER BY id"
    query += f" LIMIT {batch_size}"
    print(f"SQL QUERY  : {query}")
    filtered_df = spark.sql(query)
    return filtered_df, start_marker, end_marker, batch_size

def get_the_batch_data_gt(catalog_name, db_name, source_data_path, task_logger_table_name, batch_size):
    start_marker, end_marker = get_task_logger(catalog_name, db_name, task_logger_table_name)
    query = f"SELECT * FROM {source_data_path}"
    if start_marker and end_marker:
        query += f" WHERE {generate_filter_condition(start_marker, end_marker)}"
    query += " ORDER BY id"
    query += f" LIMIT {batch_size}"
    print(f"SQL QUERY  : {query}")
    filtered_df = spark.sql(query)
    return filtered_df, start_marker, end_marker

def generate_filter_condition(start_marker, end_marker):
    filter_column = 'id'  # Replace with the actual column name
    return f"{filter_column} > {end_marker}"

# Update Task Logger - Z-Ordered
def update_task_logger(catalog_name, db_name, task_logger_table_name, end_marker, batch_size):

    start_marker = end_marker + 1
    end_marker = end_marker + batch_size
    print(f"start_marker : {start_marker}")
    print(f"end_marker : {end_marker}")
    # Determination of table name on which markers have been calculated

    # Updating task log with new metadata
    schema = StructType(
        [
            StructField("start_marker", IntegerType(), True),
            StructField("end_marker", IntegerType(), True),
            StructField("table_name", StringType(), True),
        ]
    )
    df_column_name = ["start_marker", "end_marker", "table_name"]
    df_record = [(int(start_marker), int(end_marker), task_logger_table_name)]
    df_task = spark.createDataFrame(df_record, schema=schema)
    now = datetime.now()
    date = now.strftime("%m-%d-%Y")
    df_task = df_task.withColumn("timestamp", F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"))
    df_task = df_task.withColumn("date", F.lit(date))
    df_task = df_task.withColumn("date", F.to_date(F.col("date")))
   
    if table_already_created(catalog_name, db_name, task_logger_table_name):
        if catalog_name and catalog_name.lower() != "none":
            spark.sql(f"USE CATALOG {catalog_name}")
        # Get the maximum id from the existing logging table
        max_id = spark.sql(f"SELECT MAX(id) as max_id FROM {db_name}.{task_logger_table_name}").collect()[0].max_id
        if max_id is None:
            max_id = 0

        # Add the new id to the new record
        df_task = df_task.withColumn("id", F.lit(max_id + 1))
        df_task.createOrReplaceTempView(task_logger_table_name)
        spark.sql(f"INSERT INTO {db_name}.{task_logger_table_name} SELECT * FROM {task_logger_table_name}")
    else:
        if catalog_name and catalog_name.lower() != "none":
            spark.sql(f"USE CATALOG {catalog_name}")
         # Add the id column starting from 1 for the first record
        df_task = df_task.withColumn("id", F.lit(1))
        df_task.createOrReplaceTempView(task_logger_table_name)
        spark.sql(f"CREATE TABLE IF NOT EXISTS {db_name}.{task_logger_table_name} AS SELECT * FROM {task_logger_table_name}")
    
    return df_task

# COMMAND ----------

task_logger_table_name = f"{output_table_configs['output_1']['table']}_task_logger"

# COMMAND ----------

features_df, start_marker, end_marker, batch_size = get_the_batch_data(output_table_configs["output_1"]["catalog_name"], output_table_configs["output_1"]["schema"], input_table_paths['input_1'], task_logger_table_name)
print(start_marker)
print(end_marker)
print(batch_size)

gt_df, start_marker, end_marker = get_the_batch_data_gt(output_table_configs["output_1"]["catalog_name"], output_table_configs["output_1"]["schema"], input_table_paths['input_2'], task_logger_table_name, batch_size)

print(start_marker)
print(end_marker)
print(batch_size)

# COMMAND ----------

features_df.count()

# COMMAND ----------

features_df.display()

# COMMAND ----------

FT_DF=features_df

# COMMAND ----------

if not FT_DF.first():
  dbutils.notebook.exit("No data is available for inference, hence exiting the notebook")

# COMMAND ----------

if input_table_configs["input_1"]["catalog_name"]:
    feature_table_path = input_table_paths["input_1"]
else:
    feature_table_path = spark.sql(f"desc formatted {input_table_paths['input_1']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

if input_table_configs["input_2"]["catalog_name"]:
    gt_table_path = input_table_paths["input_2"]
else :   
    gt_table_path = spark.sql(f"desc formatted {input_table_paths['input_2']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(feature_table_path, gt_table_path)

# COMMAND ----------

catalog_name = solution_config["train"]["model_configs"]["model_registry_params"]["catalog_name"]
schema_name = solution_config["train"]["model_configs"]["model_registry_params"]["schema_name"]
model_name = solution_config["train"]["model_configs"]["model_params"]["model_name"]
Model_name = f"{catalog_name}.{schema_name}.{model_name}"

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# COMMAND ----------

mlclient.log(
    operation_type="job_run_add", 
    session_id = sdk_session_id, 
    dbutils = dbutils, 
    request_type = "inference", 
    job_config = {
        "table_name" : output_table_configs['output_1']['table'],
        "table_type" : "Inference_Output",
        "batch_size" : batch_size,
        "output_table_name" : output_table_configs['output_1']['table'],
        "model_name" : Model_name,
        "model_version" : model_version,
        "feature_table_path" : feature_table_path,
        "ground_truth_table_path" : gt_table_path,
        "env" : tracking_env,
        "quartz_cron_expression" : cron_job_schedule
    },
    spark = spark,
    tracking_env = tracking_env,
    tracking_url = tracking_url,
    verbose = True,
)

# COMMAND ----------

ground_truth = gt_df.select([input_table_configs["input_2"]["primary_keys"]] + target_columns).toPandas()
transformed_features_df = FT_DF.toPandas()
inference_df = transformed_features_df[feature_columns]
# inference_df = inference_df.rename(columns={date_column: "ds"})
display(inference_df)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{Model_name}/{model_version}"
loaded_model = mlflow.pyfunc.load_model(model_uri)
# loaded_model = mlflow.xgboost.log_model(model_uri)

# COMMAND ----------

# inference_df_np = inference_df.to_numpy()
predictions = loaded_model.predict(inference_df)
type(predictions)

# COMMAND ----------

transformed_features_df["prediction"] = predictions.round().astype(int)

transformed_features_df = pd.merge(transformed_features_df, ground_truth, on=input_table_configs["input_1"]["primary_keys"], how='inner')
output_table = spark.createDataFrame(transformed_features_df)

# COMMAND ----------

output_table.display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Load Inference Features Data

# COMMAND ----------

from MLCORE_SDK.helpers.mlc_job_helper import get_job_id_run_id

job_id, run_id ,task_id= get_job_id_run_id(dbutils)

output_table = output_table.withColumnRenamed(target_columns[0],"ground_truth_value")
output_table = output_table.withColumn("acceptance_status",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_time",F.lit(None).cast("long"))
output_table = output_table.withColumn("accepted_by_id",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_by_name",F.lit(None).cast("string"))
output_table = output_table.withColumn("moderated_value",F.lit(None).cast("double"))
output_table = output_table.withColumn("inference_job_id",F.lit(job_id).cast("string"))
output_table = output_table.withColumn("inference_run_id",F.lit(run_id).cast("string"))
output_table = output_table.withColumn("model_name",F.lit(model_name).cast("string"))
output_table = output_table.withColumn("model_version",F.lit(model_version).cast("string"))
output_table = output_table.withColumn("inference_task_id",F.lit(task_id).cast("string"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Output Table

# COMMAND ----------

output_table.display()

# COMMAND ----------

from datetime import datetime
from pyspark.sql import (
    types as DT,
    functions as F,
    Window
)
def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Save Output Table

# COMMAND ----------

db_name = output_table_configs["output_1"]["schema"]
table_name = output_table_configs["output_1"]["table"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none":
  spark.sql(f"USE CATALOG {catalog_name}")
else:
  spark.sql(f"USE CATALOG hive_metastore")

# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

output_table.createOrReplaceTempView(table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

if catalog_name:
  output_1_table_path = output_path
else:
  output_1_table_path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Features Hive Path : {output_1_table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Output Table dbfs path

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### REGISTER TASK LOGGER ON MLCORE

# COMMAND ----------

df_task = update_task_logger(output_table_configs["output_1"]["catalog_name"], output_table_configs["output_1"]["schema"],task_logger_table_name,end_marker, batch_size)

logger_table_path=f"{catalog_name}.{db_name}.{task_logger_table_name}"
if catalog_name and catalog_name.lower() != "none": 
    task_logger_table_path = logger_table_path
else:
    task_logger_table_path = spark.sql(f"desc {logger_table_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
start_marker = end_marker + 1
end_marker = end_marker + batch_size

# Register Task Logger Table in MLCore
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = task_logger_table_name,
    num_rows = df_task.count(),
    tracking_env = tracking_env,
    cols = df_task.columns,
    column_datatype = df_task.dtypes,
    table_schema = df_task.schema,
    primary_keys = ["id"],
    table_path = task_logger_table_path,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    table_sub_type="Inference_Batch",
    platform_table_type = "Task_Log",
    tracking_url = tracking_url,
    verbose=True,)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Register Inference artifacts

# COMMAND ----------

deployment_master_id=mlclient.log(operation_type = "register_inference",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    output_table_name=output_table_configs["output_1"]["table"],
    output_table_path=output_1_table_path,
    feature_table_path=feature_table_path,
    ground_truth_table_path=gt_table_path,
    model_name=Model_name,
    model_version=model_version,
    num_rows=output_table.count(),
    cols=output_table.columns,
    column_datatype = output_table.dtypes,
    table_schema = output_table.schema,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    batch_size = batch_size,
    tracking_env = tracking_env,
    compute_usage_metrics = compute_metrics,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    # register_in_feature_store=True,
    tracking_url = tracking_url,
    start_marker = start_marker,
    end_marker = end_marker,
    verbose=True)

# COMMAND ----------

if not deployment_master_id :
    dbutils.notebook.exit("Inference is not registered successfully hence skipping the saving of the data in the aggregate table.")

# COMMAND ----------

db_name = output_table_configs["output_1"]["schema"]
aggregated_table_name = f"{sdk_session_id}_inference_output_aggregated_table"
catalog_name = output_table_configs["output_1"]["catalog_name"]

if catalog_name and catalog_name.lower() != "none":
    output_path = f"{catalog_name}.{db_name}.{aggregated_table_name}"
else :
    output_path = f"{db_name}.{aggregated_table_name}"

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none":
  spark.sql(f"USE CATALOG {catalog_name}")

# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

inference_table_exists = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == aggregated_table_name.lower() and not table_data.isTemporary]

# IF the table exists, reset the ID based on existing max marker
if any(inference_table_exists):
    max_id_value = spark.sql(f"SELECT max(id) FROM {output_path}").collect()[0][0]
    window = Window.orderBy(F.monotonically_increasing_id())
    output_table = output_table.withColumn("id", F.row_number().over(window) + max_id_value)

output_table = output_table.withColumn("deployment_master_id", F.lit(deployment_master_id).cast("string"))

# Create temporary View.
output_table.createOrReplaceTempView(aggregated_table_name)

if not any(inference_table_exists):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} PARTITIONED BY (deployment_master_id) AS SELECT * FROM {aggregated_table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} PARTITION (deployment_master_id) SELECT * FROM {aggregated_table_name}")

if catalog_name and catalog_name.lower() != "none":
  aggregated_table_path = output_path
else:
  aggregated_table_path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Aggregated Inference Output Table Hive Path : {aggregated_table_path}")

# COMMAND ----------

# Register Aggregate Train Output in MLCore
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = aggregated_table_name,
    num_rows = output_table.count(),
    tracking_env = tracking_env,
    cols = output_table.columns,
    column_datatype = output_table.dtypes,
    table_schema = output_table.schema,
    primary_keys = ["id"],
    table_path = aggregated_table_path,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    table_sub_type="Inference_Output",
    platform_table_type = "Aggregated_Model_Inference_Output",
    tracking_url = tracking_url,
    verbose=True,)
