# Databricks notebook source
# MAGIC %md
# MAGIC ## INSTALL MLCORE SDK

# COMMAND ----------

# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install /Volumes/mlcore_dev/mlcore_init_scripts/mlworkspace/MLCORE_INIT/monitor_db_uc/MLCoreSDK_monitor_db_uc-0.4.6-py3-none-any.whl --force-reinstall
# MAGIC %pip install sparkmeasure

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
    with open('/Workspace/Repos/MLOpsFlow/retail_price/data_config/SolutionConfig.yaml', 'r') as solution_config:
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
import time
from sklearn.metrics import *
import mlflow
import mlflow.pyfunc
import xgboost as xgb

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

ft_data.count(), gt_data.count()

# COMMAND ----------

input_table_configs["input_1"]["primary_keys"], input_table_configs["input_2"]["primary_keys"]

# COMMAND ----------

features_data = ft_data.select([input_table_configs["input_1"]["primary_keys"]] + feature_columns)
ground_truth_data = gt_data.select([input_table_configs["input_2"]["primary_keys"]] + target_columns)

# COMMAND ----------

# DBTITLE 1,Joining Feature and Ground truth tables on primary key
final_df = features_data.join(ground_truth_data, on = input_table_configs["input_1"]["primary_keys"])
final_df = final_df.drop(input_table_configs["input_1"]["primary_keys"])

# COMMAND ----------

# DBTITLE 1,Converting the Spark df to Pandas df
final_df_pandas = final_df.toPandas()
final_df_pandas.head()

# COMMAND ----------

final_df_pandas.shape

# COMMAND ----------

# DBTITLE 1,Dropping the null rows in the final df
final_df_pandas.dropna(inplace=True)
final_df_pandas.shape

# COMMAND ----------

# DBTITLE 1,Spliting the Final df to test and train dfs
# Split the Data to Train and Test
traindf = final_df_pandas.iloc[int(final_df_pandas.shape[0] * test_size):]
testdf = final_df_pandas.iloc[:int(final_df_pandas.shape[0] * test_size)]

# COMMAND ----------

class DemandForecastingModel(mlflow.pyfunc.PythonModel):
    """
    DemandForecasting
    """
    def __init__(self):
        self.model = None
        self.pcn_encode_dict = {
            'bed_bath_table': 0,
            'garden_tools': 1,
            'consoles_games': 2,
            'health_beauty': 3,
            'cool_stuff': 4,
            'perfumery': 5,
            'computers_accessories': 6,
            'watches_gifts': 7,
            'furniture_decor': 8
        }
        self.round_cols = ["freight_price", "unit_price", "s", "comp_1", "comp_2", "comp_3", "lag_price", "avg_comp_price_by_product", "avg_customers_by_product", "avg_unit_price_by_product"]
    
    def feature_engineering(self, df):
        """Applying Feature Engineering to the input DataFrame."""
        
        df = df.copy()  # Avoid modifying original DataFrame
        
        # Drop the date column
        df.drop(columns=["date"], inplace=True)

        # Encoding categorical column
        df["product_category_name"] = df["product_category_name"].map(self.pcn_encode_dict)

        # A New Column: Avg unit_price, which will contain the mean of unit_price for each group of same month, product_category_name.
        df["avg_unit_price_by_product"] = df.groupby(["month", "product_category_name"])["unit_price"].transform("mean")

        # A New Column: Avg customers, which will contain the mean number of customers for each group of same month, product_category_name.
        df["avg_customers_by_product"] = df.groupby(["month", "product_category_name"])["customers"].transform("mean")

        # A New Column: Average Competitor Prices, which will contain the mean of 3 competitor prices for each group of same month, product_category_name.
        df["avg_comp_price_by_product"] = df.groupby(["month", "product_category_name"])[["comp_1", "comp_2", "comp_3"]].transform("mean").mean(axis=1)

        # Roudning off float value columns to 2 digits.
        for col in self.round_cols:
            df[col] = df[col].round(2)

        return df
    
    def train(self, X_train, y_train):
        """Performs feature engineering and trains the XGBoost model."""
        X_train = self.feature_engineering(X_train)
        # Train XGBoost Model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=5,
            learning_rate=0.3,
            n_estimators=500,
            reg_lambda=10,            # L2 regularization
            reg_alpha=2,              # L1 regularization
            eval_metric='rmse'
        )
        self.model.fit(X_train, y_train)
    
    def predict(self, context, X_test):
        """Applies the trained model on new data."""
        X_test = self.feature_engineering(X_test)
        # Ensure model is trained
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call `train()` first.")

        return self.model.predict(X_test)
        

# COMMAND ----------

X_train = traindf.drop(columns=target_columns)
y_train = traindf[target_columns[0]]

X_test = testdf.drop(columns=target_columns)
y_test = testdf[target_columns[0]]

# Create model instance.
model = DemandForecastingModel()

# Train the model
model.train(X_train, y_train)

# Predictions using the trained model.
y_pred_train = model.predict(context=None, X_test=X_train)
print("y_pred_train shape:", y_pred_train.shape)

y_pred = model.predict(context=None, X_test=X_test)
print("y_pred shape:", y_pred.shape)


first_row_dict = model.feature_engineering(X_train[:5]).to_numpy()

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

# MAGIC %md
# MAGIC ## SAVE PREDICTIONS TO HIVE

# COMMAND ----------

pred_train = traindf
pred_train["prediction"] = y_pred_train
pred_train["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "train"

pred_test = testdf
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
        model_name = f"{model_configs.get('model_registry_params').get('catalog_name')}.{model_configs.get('model_registry_params').get('schema_name')}.{model_configs.get('model_params').get('model_name')}"
        mlflow.set_registry_uri('databricks-uc')
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
# now = datetime.now()
# date = now.strftime("%m-%d-%Y")
train_output_df = train_output_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
# train_output_df = train_output_df.withColumn("date", F.lit(date))
# train_output_df = train_output_df.withColumn("date", to_date_(F.col("date")))

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

from mlflow.models.signature import infer_signature
model_signature = infer_signature(X_train, pred_train["prediction"].head(5))

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
    hp_tuning_result={},
    compute_usage_metrics = compute_metrics,
    taskmetrics = taskmetrics,
    stagemetrics = stagemetrics,
    tracking_env = tracking_env,
    model_documentation_url = "/Workspace/Repos/MLOpsFlow/retail_price/notebooks/model_documentation.md",
    model_configs = model_configs,
    example_input = first_row_dict,
    tracking_url = tracking_url,
    signature = model_signature,
    verbose=True,
    )

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
