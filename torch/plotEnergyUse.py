#!/usr/bin/env python
""" 
plot energy usage by PM jobs
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
from pprint import pprint
from toolbox.Plotter_Backbone import Plotter_Backbone
from toolbox.Util_IOfunc import read_one_csv
from toolbox.Util_Misc import  expand_dash_list,smoothF

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-j", "--jobId",nargs="+",default=['64975'],  help=" blank separated list of job IDs, takes n1-n2")

    parser.add_argument("-o", "--outPath", default='out/',help="output path for plots and tables")

    parser.add_argument( "--smoothWindow", default=0, type=int,help=" smooth the data using a window with requested size (bins)")
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")


    args = parser.parse_args()
    args.prjName='cosmoHpo'

    args.jobId=expand_dash_list(args.jobId)
    #PM
    
    args.sourcePath='/pscratch/sd/b/balewski/tmp_digitalMind/neuInv/benchmark/september/'
    args.formatVenue='prod'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def ana_one_job(jobId,table):
    tL=[]; eL={'node_energy':[],'cpu_energy':[],'memory_energy':[]}
    for k in range(4): eL['gpu%d_energy'%k]=[]
    for rec in table:
        #print('rec',rec)
        t=float(rec['unix_millisec'])/1000.
        tL.append(t)
        for x in eL: eL[x].append( float(rec[x]))
        #print('t',t,'eL',eL)
    
    N=len(tL)
    eL['4gpu_energy']=[]
    # sume GPU energy
    for i in range(N):
        sum=0
        for k in range(4):  sum+=eL['gpu%d_energy'%k][i]
        eL['4gpu_energy'].append(sum)


    #... convert list to NP arrays
    for x in eL: eL[x]=np.array(eL[x])
    
    if args.smoothWindow>0:
        for x in eL: eL[x]=smoothF(eL[x],args.smoothWindow)
        
    #..... convert energy to power
    tL=np.array(tL)
    tL-=tL[0]
    
    pL={x:[0] for x in eL}
    for i in range(1,N):
        dt=tL[i]-tL[i-1]
        #print(i,dt)
        for x in eL: pL[x].append( (eL[x][i]- eL[x][i-1])/dt)

    eT={}
    for x in eL:
        eT[x]= eL[x][-1]- eL[x][0]
    elaT=tL[-1]-tL[0]
    outD={'elaT':elaT,'tot_ene':eT,'jobId':jobId,'hostname':rec['hostname']}
    pprint(outD)
    outD['power']=pL
    outD['time']=tL
    return outD
    for i in range(N):
        print(i, pL['node_energy'][i], pL['4gpu_energy'][i])

        
#............................
#............................
#............................
class Plotter_EnergyUse(Plotter_Backbone):
    def __init__(self, args):
        Plotter_Backbone.__init__(self,args)

    #...!...!....................
    def one_job(self,jobD,figId=5):
        nrow,ncol=1,1
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        figId=self.smart_append(figId)
        self.plt.figure(figId,facecolor='white', figsize=(10,6))
        ax=self.plt.subplot(nrow,ncol,1)

        tit='jobId=%s, node=%s'%(jobD['jobId'],jobD['hostname'])
        T=jobD['time']
        for k in range(1,4): jobD['power'].pop('gpu%d_energy'%k)
        for name in jobD['power']:

            Y=jobD['power'][name]
            ene=jobD['tot_ene'][name] /3600.
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

stockD={}
for jobId in args.jobId:
    inpF=args.sourcePath+'%s/energy_log.csv'%jobId
    table,label=read_one_csv(inpF)
    jobD=ana_one_job(jobId,table)
plot=Plotter_EnergyUse(args)

plot.one_job(jobD)
plot.display_all('aa')

ok1
auxD={'jobId':jobId,'maxLoss':args.maxLoss}
plot.overview(ordD, title='jid='+jobId,maxLoss=args.maxLoss)

if 1:  # hpar correlations
    plot.hpar_Train(ordD,auxD)    
    #plot.hpar_FC(ordD,auxD)
    #plot.hpar_CNN(ordD,auxD)



