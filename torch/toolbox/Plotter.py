__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np

from matplotlib import cm as cmap
import matplotlib as mpl # for LogNorm()
from Plotter_Backbone import Plotter_Backbone

#...!...!..................
def get_arm_color(parName):
    armCol={'api':'C2', 'axn':'C3','som':'C4','den':'C5'}
    arm=parName.split('.')[-1]
    hcol=armCol[arm]
    return hcol

#............................
#............................
#............................
class Plotter_NeuronInverter(Plotter_Backbone):
    def __init__(self, args,inpMD,sumRec=None):
        Plotter_Backbone.__init__(self,args)
        self.maxU=1.1
        self.inpMD=inpMD
        self.sumRec=sumRec
        self.formatVenue=args.formatVenue
        
#...!...!..................
    def frames_vsTime(self,X,Y,nFr,figId=7,metaD=None, stim=[]):
        
        if metaD==None:  metaD=self.inpMD
        probeNameL=metaD['featureName']
        
        nBin=X.shape[1]
        maxX=nBin ; xtit='time bins'
        binsX=np.linspace(0,maxX,nBin)
        numProbe=X.shape[-1]
        
        assert numProbe<=len(probeNameL)
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,8))

        nFr=min(X.shape[0],nFr)
        nrow,ncol=3,3
        if nFr==1:  nrow,ncol=1,1
        print('plot input traces, numProbe',numProbe)

        yLab='ampl (a.u.)'
        
        j=0
        for it in range(nFr):
            #  grid is (yN,xN) - y=0 is at the top,  so dumm
            ax = self.plt.subplot(nrow, ncol, 1+j)
            j+=1
                        
            for ip in range(0,numProbe):
                amplV=X[it,:,ip]#/metaD2['voltsScale']                
                ax.plot(binsX,amplV,label='%d:%s'%(ip,probeNameL[ip]))
                
            if len(stim)>0:
                ax.plot(stim*10, label='stim',color='black', linestyle='--')

            tit='id%d'%(it)
            ptxtL=[ '%.2f'%x for x in Y[it]]
            tit+=' U:'+','.join(ptxtL)
            ax.set(title=tit[:45]+'..',ylabel=yLab, xlabel=xtit)
                        
            # to zoom-in activate:
            if nFr==11:
                ax.set_xlim(4000,5000)
                ax.set_ylim(-1.6,-1.4)
            ax.grid()
            if it==0:
                ax.legend(loc='best', title='input channels')

#...!...!..................
    def param_residua2D(self,U,Z, do2DRes=False, figId=9):
        #colMap=cmap.rainbow
        assert U.shape[1]==Z.shape[1]
        colMap=cmap.GnBu

        parName=self.inpMD['parName']
        nPar=self.inpMD['numPar']

        sumRec=self.sumRec
        nrow,ncol=4,5 # match to pitchfork layout 
        #nrow,ncol=4,4

        if  self.formatVenue=='poster':
            # grant August-2020
            colMap=cmap.Greys
            figId+=100

        figId=self.smart_append(figId)
        self.plt.figure(figId,facecolor='white', figsize=(14,9.))

        #1fig, axs = self.plt.subplots(nrow,ncol, sharex='col', sharey='row', gridspec_kw={'hspace': 0.3, 'wspace': 0.1},num=figId)

        fig, axs = self.plt.subplots(nrow,ncol,num=figId)
        axs=axs.flatten()
        j=0

        for iPar in range(0,nPar):
            ax1=axs[j]; j+=1
            ax1.set_aspect(1.0)
                        
            u=U[:,iPar]
            z=Z[:,iPar]

            mm2=self.maxU
            mm1=-mm2
            mm3=self.maxU/3.  # adjust here of you want narrow range for 1D residua
            binsX=np.linspace(mm1,mm2,30)

            zsum,xbins,ybins,img = ax1.hist2d(z,u,bins=binsX,#norm=mpl.colors.LogNorm(),
                                               cmin=1, cmap = colMap)

            ax1.plot([0, 1], [0,1], color='magenta', linestyle='--',linewidth=0.5,transform=ax1.transAxes) #diagonal
            # 
            ax1.set_title('%d:%s'%(iPar,parName[iPar]), size=10)

            if  self.formatVenue=='poster': continue

            # more details  will make plot more crowded
            self.plt.colorbar(img, ax=ax1)

            # .... compute residua per parameter
            umz=u-z
            resM=umz.mean()
            resS=umz.std()

            # additional info Roy+Kris did not wanted to see for nicer look
            if j>(nrow-1)*ncol: ax1.set_xlabel('pred (a.u.)')
            if j%ncol==1: ax1.set_ylabel('truth (a.u.)')

            ax1.text(0.4,0.03,'avr=%.3f\nstd=%.3f'%(resM,resS),transform=ax1.transAxes)
            #print('aa z, u, umz, umz-z',parName[iPar],z.mean(),u.mean(),umz.mean(),z.mean()-umz.mean())

            if resS > self.sumRec['lossThrHi']:
                ax1.text(0.1,0.7,'BAD',color='red',transform=ax1.transAxes)
            if resS < 0.01:
                ax1.text(0.1,0.6,'CONST',color='sienna',transform=ax1.transAxes)

            dom=self.sumRec['domain']
            tit1='dom='+dom
            tit2='MSEloss=%.3g'%self.sumRec[dom+'LossMSE']
            tit3='inp:'+str(self.sumRec['inpShape'])
            
            yy=0.90; xx=0.04
            if j==0: ax1.text(xx,yy,tit1,transform=ax1.transAxes)
            if j==1: ax1.text(xx,yy,'nSampl=%d'%(u.shape[0]),transform=ax1.transAxes)
            if j==2: ax1.text(xx,yy,tit2,transform=ax1.transAxes)
            if j==3: ax1.text(xx,yy,'short:'+self.sumRec['short_name'][:20],transform=ax1.transAxes)

            if j==6: ax1.text(0.2,yy,tit3,transform=ax1.transAxes)
            
        # more info in not used pannel
        dataTxt='data:'+sumRec['short_name'][:20]
        txt3='\ndesign:%s\n'%(sumRec['modelDesign'])+dataTxt
        txt3+='\ntrain.loss valid %.3g'%(sumRec['loss_valid'])
        txt3+='\npred.loss %s %.3g'%(sumRec['domain'],sumRec[sumRec['domain']+'LossMSE'])
        txt3+='\ninp:'+str(sumRec['inpShape'])+',  nSampl=%d'%(u.shape[0])
        txt3+='\n train ranks=%d  time/min=%.1f '%(sumRec['trainRanks'],sumRec['trainTime']/60.)
        ax1=axs[j]
        ax1.text(0.02,0.2,txt3,transform=ax1.transAxes)



#...!...!..................
    def params1D(self,P,tit1,figId=4):
        
        metaD=self.inpMD
        parName=metaD['parName']
        nPar=metaD['numPar']
        nrow,ncol=3,5
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(2.75*ncol,2.2*nrow))

        tit0=self.sumRec['short_name'][:27]
        j=1
        mm2=1.2; mm1=-mm2
        
        binsX= np.linspace(mm1,mm2,30)            
        for i in range(0,nPar):
            ax=self.plt.subplot(nrow,ncol,j)
            p=P[:,i]
            hcol=get_arm_color(parName[i])
            if 'true' in tit1: hcol='C0'

            j+=1
            (binCnt,_,_)=ax.hist(p,binsX,color=hcol)
            cntIn=sum(binCnt)
                        
            ax.set(title=parName[i], xlabel='Upar %d, inRange=%d'%(i,cntIn),ylabel='samples')
                        
            for x in [-1.,1.]:
                ax.axvline(x, color='C2', linestyle='--')
            ax.grid()
            if i==0:
                ax.text(-0.05,0.85,tit0,transform=ax.transAxes, color='r')
            if i==2:
                ax.text(0.05,0.85,tit1,transform=ax.transAxes, color='r')
            if i==1:
                ax.text(0.05,0.85,'n=%d'%p.shape[0],transform=ax.transAxes, color='r')

#...!...!..................
    def params_vs_expTime(self,P,bigD,figId=4):  # only for experimental data       
        metaD=self.inpMD
        parName=metaD['parName']
        nPar=metaD['numPar']
        nrow,ncol=3,5
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(2.6*ncol,2.2*nrow))

        tit0=self.sumRec['short_name'][:27]
        j=1
        mm2=1.2; mm1=-mm2

        sweepTA=bigD['sweep_trait']
        wallT=[]
        bbV=[]
        for rec in sweepTA:
            sweepId, sweepTime, serialRes=rec
            wallT.append(sweepTime/60.) # now in minutes         
        wallT=np.array(wallT)

        if 0:
            ix=[6,7]
            wallT=np.delete(wallT,ix)
            P=np.delete(P,ix,axis=0)
            print('skip %s measurement !!!'%str(ix),P.shape)
           
        binsX= np.linspace(mm1,mm2,30)
        for i in range(0,nPar):            
            ax=self.plt.subplot(nrow,ncol,j)
            j+=1
            uval=P[:,i]
            hcol=get_arm_color(parName[i])
            ax.plot(uval,wallT,'*-',color=hcol)            
            ax.set(title=parName[i], xlabel='pred Upar %d'%(i),ylabel='wall time (min)')
                       
            for x in [-1.,1.]:
                ax.axvline(x, color='C2', linestyle='--')
            ax.grid()
            yy=0.9
            if i==0:
                ax.text(-0.05,yy,tit0,transform=ax.transAxes, color='r')
            if i==1:
                ax.text(0.05,yy,'n=%d'%uval.shape[0],transform=ax.transAxes, color='r')

