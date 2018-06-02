import csv
import pandas as pd
import numpy as np
from  odps.df import DataFrame

class data_import:
    def __init__(self):
        #self.ads = pd.read_csv('t_ads.csv')
        #self.ads_df = DataFrame(self.ads,unknown_as_string = True)
        self.sales_sum = pd.read_csv('t_sales_sum.csv')
        self.sales_sum_df = DataFrame(self.sales_sum,unknown_as_string=True)

    def display(self):
        # print(self.ads[:3])
        # print(self.ads_df[:3])
        print(self.sales_sum_df[:3])

    def data_preprocession(self):
        self.sales_sum_df = self.sales_sum_df.select(self.sales_sum_df.exclude('dt'), dt_month=self.sales_sum_df['dt'][:7])
        self.sales_sum_df = self.sales_sum_df.groupby('shop_id','dt_month').agg(sale_amt_3m_sum = self.sales_sum_df['sale_amt_3m'].sum())

    def to_csv(self):
        df_to_csv = self.sales_sum_df.to_pandas()
        df_to_csv.to_csv('t_sales_sum_month.csv')

data = data_import()
data.data_preprocession()
data.display()
data.to_csv()

