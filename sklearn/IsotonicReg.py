import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

n=100
x=np.arange(n)
rs=check_random_state(0)
y=rs.randint(-50,50,size=(n))+50*np.log(1+np.arange(n))

model_iso=IsotonicRegression()
y_=model_iso.fit_transform(x,y)

model_lin=LinearRegression()
model_lin.fit(x[:,np.newaxis],y)

segments=[[[i,y[i]],[i,y_[i]]] for i in range(n)]
lc=LineCollection(segments,zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5*np.ones(n))

fig=plt.figure()
plt.plot(x,y,'r.',markersize=12)
plt.plot(x,y_,'g-',markersize=12)
plt.plot(x,model_lin.predict(x[:,np.newaxis]),'b-')
plt.gca().add_collection(lc)
plt.legend(('Data','Iso','Linear'),loc='lower right')
plt.title('Iso Reg')
plt.show()
