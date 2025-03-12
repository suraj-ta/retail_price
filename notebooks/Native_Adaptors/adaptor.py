from Native_Adaptors.mlcore_adaptor import read_data_mlcore



def get_monitoring_tables(
    dbutils,
    spark,
    table_details=None,
):
    return read_data_mlcore(dbutils, spark, table_details)
