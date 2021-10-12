
import pandas as pd

from ArchetypcalAnalysis import AA
from py_pcha_master.py_pcha.PCHA import PCHA

#import data
data_all=pd.read_csv(r"C:\Users\Andre\OneDrive - Danmarks Tekniske Universitet\Bachelor project\Schmidt_et_al_2021_Latent_profile_analysis_of_human_values_SUPPL\VB_LPA\Data\ESS8_data.csv")
keys=["SD1","PO1","UN1","AC1","SC1","ST1","CO1","UN2","TR1","HD1","SD2","BE1","AC2","SC2","ST2","CO2","PO2","BE2","UN3","TR2","HD2"]
data=data_all[keys]
data=data.head(100)

#A=PCHA(data.values.T,3,verbose=True)
#print(A)
AA(data.T,3)