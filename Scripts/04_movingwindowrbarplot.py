import pandas as pd
import os
import matplotlib.pyplot as plt

file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronology_stats', 'rbar50yearpbw10.ind')
pbw10rbar = pd.read_csv(file_path)
file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronology_stats', 'rbar50yearpbw20.ind')
pbw20rbar = pd.read_csv(file_path)
file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronology_stats', 'rbar50yearpbw80.ind')
pbw80rbar = pd.read_csv(file_path)

fig=plt.figure(figsize=(4, 3))
plt.step(pbw10rbar['mid.year'],pbw10rbar['rbar.eff'],color='#D75E6A',label='aMXD 10 $\mu$m',linewidth=1.5)
plt.step(pbw20rbar['mid.year'],pbw20rbar['rbar.eff'],color='#A21C57',label='aMXD 20 $\mu$m',linewidth=1.5)
plt.step(pbw80rbar['mid.year'],pbw80rbar['rbar.eff'],color='#49006a',label='aMXD 80 $\mu$m',linewidth=1.5)
plt.grid(linestyle='--',alpha=0.5)
plt.legend(frameon=False,fontsize=8,handlelength=1, columnspacing=0.5,labelspacing=0.5,
           handletextpad=0.5,ncols=2,loc='upper left') 
plt.ylim(0,0.8)
plt.xlabel('Year')
plt.ylabel('rbar')
plt.title('50-year moving window rbar',fontsize=10)
plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures/movingrbar.eps'), format='eps',bbox_inches='tight')
plt.show()