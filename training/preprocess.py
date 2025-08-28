import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.sql.functions import count,sum,col,when,mean
from pyspark.ml.feature import Imputer, StringIndexer,OneHotEncoder,VectorAssembler
from .utils import ColumnDropper,MasterColumnCreator,GenderFiller,AgeImputer

def preprocessing_pipeline():

    master_column_creator=MasterColumnCreator()
    num_cat_imputer=Imputer(inputCols=["Pclass","SibSp","Parch"],
                    outputCols=["Pclass","SibSp","Parch"],
                    strategy="median")

    num_imputer=Imputer(inputCols=["Age","Fare"],
                    outputCols=["Age","Fare"],
                    strategy="mean")

    age_imputer=AgeImputer()
    gender_imputer=GenderFiller()
    dropper = ColumnDropper(columns_to_drop=["PassengerId","Name","Ticket","Cabin","Embarked"]) 

    string_indexer=StringIndexer(inputCol='Sex',
                                outputCol='idx_Sex'
    )

    ohe_encoder=OneHotEncoder(inputCol='idx_Sex',
                                outputCol='ohe_sex'
    )

    vector_assembler=VectorAssembler(inputCols=['Pclass',"ohe_sex","Age","SibSp","Parch",'Fare'],
                                    outputCol='features'
    )

    preprocessing=Pipeline(stages=[master_column_creator,
                            num_cat_imputer,
                            num_imputer,
                            age_imputer,
                            gender_imputer,
                            dropper,
                            string_indexer,
                            ohe_encoder,
                            vector_assembler,
                            ])

    return preprocessing