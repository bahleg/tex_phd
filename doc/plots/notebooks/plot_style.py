# -*- coding: utf-8 -*-
from matplotlib import pylab as plt
def init_rc_params():
	plt.rcParams['font.family'] = 'serif'
	plt.rcParams['font.serif'] = 'FreeSerif' 
	plt.rcParams['lines.linewidth'] = 2
	plt.rcParams['lines.markersize'] = 35
	#plt.rcParams['scatter.markersize'] = 35
	plt.rcParams['xtick.labelsize'] = 24
	plt.rcParams['ytick.labelsize'] = 24
	plt.rcParams['legend.fontsize'] = 24	
	plt.rcParams['axes.titlesize']=36
	plt.rcParams['axes.labelsize']=24
	
if __name__=='__main__':
	init_rc_params()
	plt.plot(range(100), range(100), label=u'Легенда')
	#plt.scatter(range(100), range(100))
	plt.title(u'Заголовок')
	plt.xlabel(u'Ось X')
	plt.ylabel(u'Ось Y')
	plt.legend(loc='best')
	plt.show()
	
