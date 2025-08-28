from pyspark.ml import Transformer, Estimator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import col, mean, when
from pyspark.sql import DataFrame

class ColumnDropper(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, columns_to_drop=None):
        super().__init__()
        self.columns_to_drop = columns_to_drop or []

    def _transform(self, df):
        return df.drop(*self.columns_to_drop)
 
class MasterColumnCreator(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, column_to_add="Master"):
        super().__init__()
        self.column_to_add = column_to_add

    def _transform(self, df):
        return df.withColumn(
            self.column_to_add,
            when(col("Name").contains("Master."), 1).otherwise(0)
        )    

class GenderFiller(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="Name", outputCol="Sex"):
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, df):
        return df.withColumn(
            self.outputCol,
            when(
                col(self.outputCol).isNull() &
                (col(self.inputCol).contains("Mr.") | col(self.inputCol).contains("Master.")),
                "male"
            ).when(
                col(self.outputCol).isNull(),
                "female"
            ).otherwise(col(self.outputCol))
        )
    
class AgeImputerModel(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, master_mean=None, female_mean=None, male_mean=None, inputCol="Age", outputCol="Age"):
        super().__init__()
        self.master_mean = master_mean
        self.female_mean = female_mean
        self.male_mean = male_mean
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            self.outputCol,
            when(col(self.inputCol).isNull() & (col('Master')==1), self.master_mean)
            .when(col(self.inputCol).isNull() & (col("Sex")=="female"), self.female_mean)
            .when(col(self.inputCol).isNull() & (col("Sex")=="male"), self.male_mean)
            .otherwise(col(self.inputCol))
        )

class AgeImputer(Estimator, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="Age", outputCol="Age"):
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _fit(self, df: DataFrame):
        # Compute means on training data
        master_mean = df.filter(col('Master')==1).select(mean(self.inputCol)).collect()[0][0]
        female_mean = df.filter(col('Sex')=='female').select(mean(self.inputCol)).collect()[0][0]
        male_mean = df.filter(col('Sex')=='male').select(mean(self.inputCol)).collect()[0][0]

        # Return a Transformer model with learned means
        return AgeImputerModel(
            master_mean=master_mean,
            female_mean=female_mean,
            male_mean=male_mean,
            inputCol=self.inputCol,
            outputCol=self.outputCol
        )
