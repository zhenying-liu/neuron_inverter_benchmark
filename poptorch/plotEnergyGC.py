#!/usr/bin/env python
""" 
plot energy usage by PM jobs
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
from pprint import pprint
#from Plotter_Backbone import Plotter_Backbone
from toolbox.Util_IOfunc import read_one_csv
import sys,os
sys.path.append(os.path.abspath("../torch/toolbox"))
from scipy import interpolate
from Util_Misc import smoothF,mini_plotter

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--dataName",default="LBL_power_160_epochs",  help="dataName")

    parser.add_argument("-o", "--outPath", default='out/',help="output path for plots and tables")

    parser.add_argument( "--smoothWindow", default=0, type=int,help=" smooth the data using a window with requested size (bins)")
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")


    args = parser.parse_args()
    args.prjName='eneGC'
   
    args.sourcePath='data-sept17/'
 
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def ana_one_job(table):
    tL=[]; pL=[]
    for rec in table:
        #print('rec',rec)
        t=float(rec['unix_sec'])
        tL.append(t)
        pw=float(rec['power_uW'])*1e-6
        pL.append(pw)
        #print('t',t,'pL',pL)

    N=len(tL)
    #... convert list to NP arrays
    tL=np.array(tL)
    pL=np.array(pL)

    #... integrate power w/ 1 sec step
    tL=tL-tL[0]
    f=interpolate.interp1d(tL, pL)
    tS=np.arange(0,tL[-1])
    pS=f(tS)
    totE=np.sum(pS)
    elaT=tL[-1]-tL[0]
    print('N=%d  duration=%.1f sec  tot_ene=%.1f (J) = %.1f (Wh) '%(N,elaT,totE,totE/3600.))
    outD={'elaT':elaT,'tot_ene':totE}
    
    if args.smoothWindow>0:
       pL=smoothF(pL,args.smoothWindow)

    pprint(outD)
    outD['4IPU']=pL
    outD['time']=tL
    return outD
        
#...!...!....................
def plot_one_job(plot,jobD):
        nrow,ncol=1,1
        plt.figure(1,facecolor='white', figsize=(10,6))
        ax=plt.subplot(nrow,ncol,1)

        tit='job='+args.dataName
        T=jobD['time']
        name='4IPU'
        if 1:
            Y=jobD[name]
            ene=jobD['tot_ene'] /3600.
            dLab='%s: %.1f'%(name,ene)
            #print(T,Y)
            ax.plot(T,Y,label=dLab)
           
        ax.legend(loc='best', title='total used energy: (Wh)')
        ax.set(xlabel='wall time (sec)',ylabel='power (W)', title=tit)
        ax.grid(True)
        return
        #if j==0: ax.text(0.1,0.85,'n=%d'%len(lossV),transform=ax.transAxes)
         
                
         

#=================================
#=================================
#  M A I N 
#=================================
#=================================
args=get_parser()

inpF=args.sourcePath+'%s.txt'%args.dataName
table,label=read_one_csv(inpF,delim=' ')
jobD=ana_one_job(table)

plt=mini_plotter(args)

plot_one_job(plt,jobD)
outF=args.dataName+'.png'
plt.savefig(outF)
print('saved:',outF)
plt.show()

