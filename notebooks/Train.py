# Databricks notebook source
# MAGIC %md
# MAGIC ## INSTALL MLCORE SDK

# COMMAND ----------

# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install /Volumes/mlcore_dev/mlcore_init_scripts/mlworkspace/MLCORE_INIT/monitor_db_uc/MLCoreSDK_monitor_db_uc-0.4.6-py3-none-any.whl --force-reinstall
# MAGIC %pip install sparkmeasure
# MAGIC # %pip install numpy==1.19.1
# MAGIC
# MAGIC # %pip install pandas==1.0.5

# COMMAND ----------

from sparkmeasure import StageMetrics, TaskMetrics

taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
from MLCORE_SDK import mlclient
from pyspark.sql import functions as F
import json

try:
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = json.loads(solution_config)
    print("Loaded config from dbutils")
except Exception as e:
    print(e)
    with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

try:
    retrain_params = (dbutils.widgets.get("retrain_params"))
    retrain_params = json.loads(retrain_params)
    print("Loaded Retrain Params from job params")
    is_retrain = True
except:
    is_retrain = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## PERFORM MODEL TRAINING 

# COMMAND ----------

# DBTITLE 1,Imports
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from prophet import Prophet
import time
from sklearn.metrics import *

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# DBTITLE 1,Input from the user
# GENERAL PARAMETERS
tracking_env = solution_config["general_configs"]["tracking_env"]
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]
 
tracking_url = solution_config["general_configs"].get("tracking_url", None)
tracking_url = f"https://{tracking_url}" if tracking_url else None

# JOB SPECIFIC PARAMETERS
input_table_configs = solution_config["train"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["train"]["datalake_configs"]['output_tables']
model_configs = solution_config["train"]["model_configs"]
storage_configs = solution_config["train"]["storage_configs"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
test_size = solution_config['train']["test_size"]
date_column = solution_config['train']["date_column"]
horizon = solution_config['train']["horizon"]
frequency = solution_config['train']["frequency"]

# COMMAND ----------

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

ft_data = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")
gt_data = spark.sql(f"SELECT * FROM {input_table_paths['input_2']}")

# COMMAND ----------

gt_data.display()

# COMMAND ----------

ft_data.count(), gt_data.count()

# COMMAND ----------

input_table_configs["input_1"]["primary_keys"]

# COMMAND ----------

features_data = ft_data.select([input_table_configs["input_1"]["primary_keys"]] + feature_columns + [date_column])
ground_truth_data = gt_data.select([input_table_configs["input_2"]["primary_keys"]] + target_columns)

# COMMAND ----------

# DBTITLE 1,Joining Feature and Ground truth tables on primary key
final_df = features_data.join(ground_truth_data, on = input_table_configs["input_1"]["primary_keys"])

# COMMAND ----------

# DBTITLE 1,Converting the Spark df to Pandas df
final_df_pandas = final_df.toPandas()
final_df_pandas.head()

# COMMAND ----------

final_df_pandas.shape

# COMMAND ----------

# DBTITLE 1,Dropping the null rows in the final df
final_df_pandas.dropna(inplace=True)

# COMMAND ----------

final_df_pandas.shape

# COMMAND ----------

# DBTITLE 1,Spliting the Final df to test and train dfs
# Split the Data to Train and Test
traindf = final_df_pandas.iloc[int(final_df_pandas.shape[0] * test_size):]
testdf = final_df_pandas.iloc[:int(final_df_pandas.shape[0] * test_size)]

# COMMAND ----------

try :
    if is_retrain:
        hyperparameters = retrain_params.get("hyperparameters", {})
        print(f"Retraining model with hyper parameters: {hyperparameters}")
        hp_tuning_result = {}
    else:
        hp_tuning_result = dbutils.notebook.run(
            "Hyperparameter_Tuning", 
            timeout_seconds=0,)
        hyperparameters = json.loads(hp_tuning_result)["best_hyperparameters"]
        report_path = json.loads(hp_tuning_result)["report_path"]
        print(f"Training Hyperparameters: {hyperparameters}")
        print(f"Report path: {report_path}")
except Exception as e:
    print(e)
    print("Using default hyper parameters")
    hyperparameters = {}
    hp_tuning_result = {}

# COMMAND ----------

if not hyperparameters or hyperparameters == {} :
    model = Prophet()
    print(f"Using model with default hyper parameters")
else :
    model = Prophet(**hyperparameters)
    print(f"Using model with custom hyper parameters")

# COMMAND ----------

for feature in feature_columns:
    model.add_regressor(feature, standardize=False)

# COMMAND ----------

train_prophet_df = traindf.rename(columns={date_column: "ds", target_columns[0]: "y"})
test_prophet_df = testdf.rename(columns={date_column: "ds", target_columns[0]: "y"})

# COMMAND ----------

train_prophet_df

# COMMAND ----------

test_prophet_df

# COMMAND ----------

X_train_np = train_prophet_df.to_numpy()
X_test_np = test_prophet_df.to_numpy()
first_row_dict = train_prophet_df[:5].to_numpy()

# COMMAND ----------

model = model.fit(train_prophet_df, iter=200)

# COMMAND ----------

y_train = traindf[target_columns[0]]
y_test = testdf[target_columns[0]]

# Predict
train_pred = model.predict(train_prophet_df)
test_pred = model.predict(test_prophet_df)

# COMMAND ----------

model.plot_components(train_pred)

# COMMAND ----------

y_pred_train = train_pred["yhat"].to_numpy()
y_pred = test_pred["yhat"].to_numpy()

# COMMAND ----------

# Predict it on Test and calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# COMMAND ----------

test_metrics = {"r2":r2, "mse":mse, "mae":mae, "rmse":rmse}
test_metrics

# COMMAND ----------

# Predict it on Train and calculate metrics
r2 = r2_score(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train, squared=False)

# COMMAND ----------

train_metrics = {"r2":r2, "mse":mse, "mae":mae, "rmse":rmse}
train_metrics

# COMMAND ----------

pred_train = traindf
pred_train["prediction"] = y_pred_train

pred_test = testdf
pred_test["prediction"] = y_pred

# COMMAND ----------

pred_test[target_columns[0]] = y_test
pred_train[target_columns[0]] = y_train

# COMMAND ----------

columns_for_reports = ["trend", "daily", "yearly", "weekly"]

# COMMAND ----------

if "daily" not in train_pred.columns:
    columns_for_reports.remove("daily")

if "weekly" not in train_pred.columns:
    columns_for_reports.remove("weekly")

if "yearly" not in train_pred.columns:
    columns_for_reports.remove("yearly")

print(f"Columns considered for report : {columns_for_reports}")

# COMMAND ----------

for report_column in columns_for_reports:
    if report_column in train_pred.columns:
        pred_train[report_column] = train_pred[report_column].to_numpy()
        pred_test[report_column] = test_pred[report_column].to_numpy()

# COMMAND ----------

from MLCORE_SDK.helpers.mlc_helper import get_job_id_run_id
job_id, run_id, task_id = get_job_id_run_id(dbutils)
print(job_id, run_id)
report_directory = f'{env}/media_artifacts/2a3b88f5bb6444b0a19e23e4ef21495a/Solution_configs_upgrades/{job_id}/{run_id}/Tuning_Trails'

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAVE PREDICTIONS TO HIVE

# COMMAND ----------

pred_train["prediction"] = y_pred_train
pred_train["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "train"

pred_test["prediction"] = y_pred
pred_test["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "test"

# COMMAND ----------

final_train_output_df = pd.concat([pred_train, pred_test])
train_output_df = spark.createDataFrame(final_train_output_df)

# COMMAND ----------

from mlflow.tracking import MlflowClient
def get_latest_model_version(model_configs):
    try : 
        mlflow_uri = model_configs.get("model_registry_params").get("host_url")
        model_name = model_configs.get("model_params").get("model_name")
        mlflow.set_registry_uri(mlflow_uri)
        client = MlflowClient()
        x = client.get_latest_versions(model_name)
        model_version = x[0].version
        return model_version
    except Exception as e :
        print(f"Exception in {get_latest_model_version} : {e}")
        return 0

# COMMAND ----------

model_name = f"{model_configs.get('model_registry_params').get('catalog_name')}.{model_configs.get('model_registry_params').get('schema_name')}.{model_configs.get('model_params').get('model_name')}"
model_name

# COMMAND ----------

from MLCORE_SDK.helpers.mlc_helper import get_job_id_run_id
job_id, run_id ,task_id= get_job_id_run_id(dbutils)
print(job_id, run_id)

# COMMAND ----------

model_version = get_latest_model_version(model_configs) + 1
train_output_df = train_output_df.withColumn("model_name", F.lit(model_name).cast("string"))
train_output_df = train_output_df.withColumn("model_version", F.lit(model_version).cast("string"))
train_output_df = train_output_df.withColumn("train_job_id", F.lit(job_id).cast("string"))
train_output_df = train_output_df.withColumn("train_run_id", F.lit(run_id).cast("string"))
train_output_df = train_output_df.withColumn("train_task_id",F.lit(task_id).cast("string"))

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats = (
        "MM-dd-yyyy",
        "dd-MM-yyyy",
        "MM/dd/yyyy",
        "yyyy-MM-dd",
        "M/d/yyyy",
        "M/dd/yyyy",
        "MM/dd/yy",
        "MM.dd.yyyy",
        "dd.MM.yyyy",
        "yyyy-MM-dd",
        "yyyy-dd-MM",
    )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

# DBTITLE 1,Adding Timestamp and Date Features to a Source 1
now = datetime.now()
date = now.strftime("%m-%d-%Y")
train_output_df = train_output_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
train_output_df = train_output_df.withColumn("date", F.lit(date))
train_output_df = train_output_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in train_output_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  train_output_df = train_output_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

db_name = output_table_configs["output_1"]["schema"]
table_name = output_table_configs["output_1"]["table"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none":
  spark.sql(f"USE CATALOG {catalog_name}")


# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

train_output_df.createOrReplaceTempView(table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

if catalog_name and catalog_name.lower() != "none":
  output_1_table_path = output_path
else:
  output_1_table_path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Features Hive Path : {output_1_table_path}")

# COMMAND ----------

if input_table_configs["input_1"]["catalog_name"]:
    feature_table_path = input_table_paths["input_1"]
else:
    feature_table_path = spark.sql(f"desc formatted {input_table_paths['input_1']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

if input_table_configs["input_2"]["catalog_name"]:
    gt_table_path = input_table_paths["input_2"]
else:
    gt_table_path = spark.sql(f"desc formatted {input_table_paths['input_2']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(feature_table_path, gt_table_path)

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

train_data_date_dict = {
    "feature_table" : {
        "ft_start_date" : ft_data.select(F.min("timestamp")).collect()[0][0],
        "ft_end_date" : ft_data.select(F.max("timestamp")).collect()[0][0]
    },
    "gt_table" : {
        "gt_start_date" : gt_data.select(F.min("timestamp")).collect()[0][0],
        "gt_end_date" : gt_data.select(F.max("timestamp")).collect()[0][0]        
    }
}

# COMMAND ----------

model_name = f"{model_configs.get('model_registry_params').get('catalog_name')}.{model_configs.get('model_registry_params').get('schema_name')}.{model_configs.get('model_params').get('model_name')}"
model_name

# COMMAND ----------

mlclient.log(
    operation_type="job_run_add", 
    session_id = sdk_session_id, 
    dbutils = dbutils, 
    request_type = "train", 
    job_config = 
    {
        "table_name" : output_table_configs["output_1"]["table"],
        "model_name" : model_name,
        "feature_table_path" : feature_table_path,
        "ground_truth_table_path" : gt_table_path,
        "feature_columns" : feature_columns,
        "target_columns" : target_columns,
        "model" : model_name,
        "model_runtime_env" : "python",
        "reuse_train_session" : False
    },
    tracking_env = tracking_env,
    tracking_url = tracking_url,
    spark = spark,
    verbose = True,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## REGISTER MODEL IN MLCORE

# COMMAND ----------

from MLCORE_SDK import mlclient

# COMMAND ----------

train_prophet_df = train_prophet_df.drop(['index', 'y'], axis=1)

# COMMAND ----------

import mlflow 
from mlflow.models.signature import infer_signature
model_signature = infer_signature(train_prophet_df, pred_train["prediction"].head(5))

# COMMAND ----------

# DBTITLE 1,Registering the model in MLCore
model_artifact_id=mlclient.log(operation_type = "register_model",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    model = model,
    model_name = model_name,
    model_runtime_env = "python",
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    feature_table_path = feature_table_path,
    ground_truth_table_path = gt_table_path,
    train_output_path = output_1_table_path,
    train_output_rows = train_output_df.count(),
    train_output_cols = train_output_df.columns,
    table_schema=train_output_df.schema,
    column_datatype = train_output_df.dtypes,
    feature_columns = feature_columns,
    target_columns = target_columns,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    train_data_date_dict = train_data_date_dict,
    hp_tuning_result=hp_tuning_result,
    compute_usage_metrics = compute_metrics,
    taskmetrics = taskmetrics,
    stagemetrics = stagemetrics,
    tracking_env = tracking_env,
    date_column=date_column,
    horizon=horizon,
    model_documentation_url = "/Workspace/Repos/MLOpsFlow/demand_forecasting_usecase/notebooks/model_documentation.md",
    frequency=frequency,
    # register_in_feature_store=True,
    model_configs = model_configs,
    example_input = first_row_dict,
    tracking_url = tracking_url,
    signature = model_signature,
    verbose = True)

# COMMAND ----------

if not model_artifact_id :
    dbutils.notebook.exit("Model is not registered successfully hence skipping the saving of tuning trials plots.")

# COMMAND ----------

db_name = output_table_configs["output_1"]["schema"]
aggregated_table_name = f"{sdk_session_id}_train_output_aggregated_table"
catalog_name = output_table_configs["output_1"]["catalog_name"]
if catalog_name and catalog_name.lower() != "none":
    output_path = f"{catalog_name}.{db_name}.{aggregated_table_name}"
else :
    output_path = f"{db_name}.{aggregated_table_name}"

# Add Model Artifact ID retrieved after registering the model.
train_output_df = train_output_df.withColumn("model_artifact_id", F.lit(model_artifact_id))

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none":
  spark.sql(f"USE CATALOG {catalog_name}")

# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

train_table_exists = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == aggregated_table_name.lower() and not table_data.isTemporary]

# IF the table exists, reset the ID based on existing max marker
if any(train_table_exists):
    max_id_value = spark.sql(f"SELECT max(id) FROM {output_path}").collect()[0][0]
    window = Window.orderBy(F.monotonically_increasing_id())
    train_output_df = train_output_df.withColumn("id", F.row_number().over(window) + max_id_value)

# Create temporary View.
train_output_df.createOrReplaceTempView(aggregated_table_name)

if not any(train_table_exists):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} PARTITIONED BY (model_artifact_id) AS SELECT * FROM {aggregated_table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} PARTITION (model_artifact_id) SELECT * FROM {aggregated_table_name}")

if catalog_name and catalog_name.lower() != "none":
  aggregated_table_path = output_path
else:
  aggregated_table_path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Aggregated Train Output Hive Path : {aggregated_table_path}")

# COMMAND ----------

# Register Aggregate Train Output in MLCore
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = aggregated_table_name,
    num_rows = train_output_df.count(),
    tracking_env = tracking_env,
    cols = train_output_df.columns,
    column_datatype = train_output_df.dtypes,
    table_schema = train_output_df.schema,
    primary_keys = ["id"],
    table_path = aggregated_table_path,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    table_sub_type="Train_Output",
    tracking_url = tracking_url,
    platform_table_type = "Aggregated_train_output",
    verbose=True,)

# COMMAND ----------

if not is_retrain:
    from MLCORE_SDK.helpers.mlc_helper import get_job_id_run_id
    job_id, run_id ,task_id= get_job_id_run_id(dbutils)
    print(job_id, run_id)

# COMMAND ----------

if not is_retrain:    
    from MLCORE_SDK.sdk.manage_sdk_session import get_session
    existing_session_data = get_session(
        sdk_session_id,
        dbutils,
        api_endpoint=tracking_url,
        tracking_env=tracking_env,
    ).json()["data"]
    existing_session_state = existing_session_data.get("state_dict", {})
    project_id = existing_session_state.get("project_id", "")
    version = existing_session_state.get("version", "")
    print(project_id, version)

# COMMAND ----------

if not is_retrain:    
    try:
        print(model_artifact_id)
        if storage_configs["cloud_provider"] == "databricks_uc":
            params = storage_configs.get("params",{})
            catalog_name=params.get("catalog_name","")
            schema_name = params.get("schema_name","")
            volume_name = params.get("volume_name","")

            artifact_path_uc_volume = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{tracking_env}/media_artifacts/{project_id}/{version}/{job_id}/{run_id}"
            print(artifact_path_uc_volume)
            mlclient.log(
                operation_type = "upload_blob_to_cloud",
                blob_path=report_path,
                dbutils = dbutils ,
                target_path = f"{artifact_path_uc_volume}/Model_Evaluation/Tuning_Trails_report_{int(time.time())}.png",
                resource_type = "databricks_uc",
                project_id = project_id,
                version = version,
                job_id = job_id,
                run_id = run_id,
                model_artifact_id = model_artifact_id,
                request_type = "Model_Evaluation",
                storage_configs = storage_configs,
                api_endpoint=tracking_url,
                tracking_env = tracking_env,
                verbose=True)
        else :
            report_directory = f"{tracking_env}/media_artifacts/{project_id}/{version}/{job_id}/{run_id}"
            print(report_directory)
            container_name = storage_configs.get("container_name")
            mlclient.log(
                operation_type = "upload_blob_to_cloud",
                source_path=report_path,
                dbutils = dbutils ,
                target_path = f"{report_directory}/Model_Evaluation/Tuning_Trails_report_{int(time.time())}.png",
                resource_type = "az",
                project_id = project_id,
                version = version,
                job_id = job_id,
                run_id = run_id,
                model_artifact_id = model_artifact_id,
                request_type = "Model_Evaluation",
                storage_configs = storage_configs,
                api_endpoint=tracking_url,
                tracking_env = tracking_env,
                verbose=True)
    except Exception as e:
        print(Exception, e)
        