from pyspark.sql import SparkSession

def spark_session_creator():
    spark = (SparkSession.builder
            .appName("ML_Training")
            .config("spark.executor.instances", "4")   # number of executors
            .config("spark.executor.cores", "2")       # cores per executor
            .config("spark.driver.memory", "2g")       # memory for driver
            .config("spark.executor.memory", "2g")     # memory per executor
            .config("spark.default.parallelism", "8")
            .getOrCreate())
    
    print('spark session created')       
    return spark
