from datetime import datetime
from heat_index import *
import pandas as pd

def float_nan(x):
	try: return float(x)
	except: return float('nan')

def datetime_index(dt):
	return datetime.fromisoformat(dt)

def read_csv(csvfile):
	data = pd.read_csv(csvfile,
	              low_memory=False,
	              header=1,
	              date_parser=datetime.fromisoformat,
	              index_col=0,
	              usecols=[1,43,48],
	              converters={1:datetime_index,43:float_nan,48:float_nan},
	              names=['datetime','temperature','humidity']).dropna()
	return heat_index(data=data,temperature="temperature",humidity="humidity")
