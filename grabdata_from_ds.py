from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///mta.db")
all_data = pd.read_sql('SELECT * FROM mta_data;', engine)
all_data = all_data.drop(index=[0,1])
print(all_data.head())


