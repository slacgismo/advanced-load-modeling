import numpy as np
import pandas as pd
from datetime import datetime
from math import *

def heat_index(data=None,temperature=None,humidity=None):
	"""
	Compute heat index from temperature and humidity

	Parameters:

		temperature (float or list)		temperature (in degF)

		humidity (float or list)		humidity (in % or fractional)

	Returns:
		float or list	heat index

	The calculation is based on the NOAA regression 
	(see https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml).

	If the inputs are lists of the same length, the each pair is converted
	and the resulting list if returned.
	"""
	if type(data) is pd.DataFrame and type(temperature) is str and type(humidity) is str:
		T = np.array(data[temperature])
		RH = np.array(data[humidity])
		data['heatindex'] = heat_index(temperature=list(T),humidity=list(RH))
		return data
	elif type(temperature) in (list,np.array) and type(humidity) in (list,np.array) and len(temperature) == len(humidity):
		HI = []
		for T,H in list(zip(temperature,humidity)):
			HI.append(heat_index(temperature=T,humidity=H))
		return HI
	elif type(temperature) in (int,float,np.float64) and type(humidity) in (int,float,np.float64):
		T = temperature
		T2 = T*T
		RH = humidity
		if RH <= 1.0: RH *= 100
		RH2 = RH*RH
		if T < 80:
			HI = 0.75*T + 0.25*( 61.0+1.2*(T-68.0)+0.094*RH)
		else:
			HI = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH - 0.00683783*T2 - 0.05481717*RH2 + 0.00122874*T2*RH + 0.00085282*T*RH2 - 0.00000199*T2*RH2
			if RH < 13 and T < 112:
				HI -= ((13-RH)/4)*sqrt((17-abs(T-95.))/17)
			elif RH > 85 and T < 87:
				HI += ((RH-85)/10) * ((87-T)/5)
			elif T > 112:
				HI = float('nan')
		return round(HI,1)
	else:
		raise Exception("temperature (type %s) or humidity (type %s) type error"%(type(temperature),type(humidity)))

def selftest():
	"""Perform a test of heat index calculations"""
	T = [80,86,92,98]
	RH = [40,50,60,70]
	HI = [80,88,105,135]
	ans = heat_index(temperature=T,humidity=RH)
	for n in range(0,len(T)):
		if abs(HI[n]-ans[n]) > 1.0:
			print(f"Test {n+1} failed: heat_index(temperature={T[n]},humidity={RH[n]}) == {HI[n]} <> {ans[n]}")
	def float_nan(x):
		try: return float(x)
		except: return float('nan')
	def datetime_index(dt):
		return datetime.fromisoformat(dt)
	SFO = pd.read_csv('../data/SFO.csv',
                  low_memory=False,
                  header=1,
                  date_parser=datetime.fromisoformat,
                  index_col=0,
                  usecols=[1,43,48],
                  converters={1:datetime_index,43:float_nan,48:float_nan},
                  names=['datetime','temperature','humidity']).dropna()
	print(heat_index(data=SFO,temperature="temperature",humidity="humidity"))

if __name__ == '__main__':
	selftest()

