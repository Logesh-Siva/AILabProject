from pyspark.ml import Transformer
from pyspark.sql.functions import count,sum,col,when,mean
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class ColumnDropper(Transformer):
    def __init__(self, columns_to_drop):
        super(ColumnDropper, self).__init__()
        self.columns_to_drop = columns_to_drop
    
    def _transform(self, df):
        return df.drop(*self.columns_to_drop)
    
class MasterColumnCreator(Transformer):
    def __init__(self, column_to_add="Master"):
        super(MasterColumnCreator, self).__init__()
        self.column_to_add = column_to_add
    
    def _transform(self, df):

        return df.withColumn(self.column_to_add,when(col("Name").contains("Master."),1).otherwise(0))
class GenderFiller(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="Name", outputCol="Sex"):
        super(GenderFiller, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, df):
        return df.withColumn(
            self.outputCol,
            when(
                col(self.outputCol).isNull() & (col(self.inputCol).contains("Mr.") | col(self.inputCol).contains("Master.")),
                "male"
            ).when(
                col(self.outputCol).isNull(),
                "female"
            ).otherwise(col(self.outputCol))
        )
    
class AgeImputer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="Name", outputCol="Age"):
        super(AgeImputer, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, df):
        return df.withColumn(
            self.outputCol,
            when(
                col(self.outputCol).isNull() & (col('Master')==1),
                df.filter(col('Master')==1).select(mean('Age')).collect()[0][0]
            ).when(
                col(self.outputCol).isNull() & (col("Sex")=="female"),
                df.filter(col('Sex')=='female').select(mean('Age')).collect()[0][0]
            ).when(
                col(self.outputCol).isNull() & (col("Sex")=="male"),
                df.filter(col('Sex')=='male').select(mean('Age')).collect()[0][0]
            ).otherwise(col(self.outputCol))
        )
