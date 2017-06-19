from pandas_datareader import data as pdr
import fix_yahoo_finance
from datetime import date

start=date(2000,1,1)
end=datetime.now()

data=pdr.get_data_yahoo('AAPL',start,end)