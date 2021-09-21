#!/usr/bin/env python
""" 
read trained net : model+weights
read test data from HD5
infere for  test data 

Inference works alwasy on 1  IPU
./predict_one.py -m outY -X

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import torch

import  time
import sys,os
import logging

from toolbox.Model import MyModelWithLoss
from toolbox.Util_IOfunc import read_yaml, write_yaml, restore_checkpoint

from toolbox.Dataloader_h5 import get_data_loader

import poptorch
import popdist.poptorch

sys.path.append(os.path.relpath("../torch/toolbox/"))
from Plotter import Plotter_NeuronInverter
import argparse

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--facility", default='corigpu', type=str)
    parser.add_argument('--venue', dest='formatVenue', choices=['prod','poster'], default='prod',help=" output quality/arangement")

    parser.add_argument("-m","--modelPath",  default='out/', help="trained model ")
    parser.add_argument("-o", "--outPath", default='same',help="output path for plots and tables")
 
    parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=False, help="disable X-term for batch mode")

    parser.add_argument("-n", "--numSamples", type=int, default=None, help="limit samples to predict")
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--cellName", type=str, default=None, help="alternative cell shortName ")
    args = parser.parse_args()
    args.prjName='neurInfer'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def load_model4infer(sumMD,modelPath):
    # ... assemble model

    device = torch.device("cuda")
    # load entirel model
    modelF = os.path.join(modelPath, sumMD['train_params']['blank_model'])
    stateF= os.path.join(modelPath, sumMD['train_params']['checkpoint_name'])

    print('M: load model:',modelF)
    myModel = torch.load(modelF)
    modelWloss=MyModelWithLoss(myModel)
    print('M: tmp popOpt re-init')
    popOpts = popdist.poptorch.Options()
    popOpts.deviceIterations(1)
    cachePath='./exec_cache'
    popOpts.enableExecutableCaching(cachePath)
    
    print("\n-----------  restore model for inference, state= ",stateF)
    startEpoch=restore_checkpoint( stateF, modelWloss)

    model4infer = poptorch.inferenceModel(modelWloss.eval(), options=popOpts)
    return model4infer,popOpts

#...!...!..................
def model_infer(model,test_loader,sumMD):

    criterion =torch.nn.MSELoss() # Mean Squared Loss
    test_loss = 0

    # prepare output container, Thorsten's idea
    num_samp=len(test_loader.dataset)
    outputSize=sumMD['train_params']['model']['outputSize']
    print('predict for num_samp=',num_samp,', outputSize=',outputSize)
    # clever list-->numpy conversion, Thorsten's idea
    Uall=np.zeros([num_samp,outputSize],dtype=np.float32)
    Zall=np.zeros([num_samp,outputSize],dtype=np.float32)
    nEve=0
    nStep=0
    cpuLossF=torch.nn.MSELoss(reduction='none' )#returns a loss per element
    
    for j,(data, target) in enumerate(test_loader):
        pred, loss_op = model4infer(data, target)
        loss=np.mean(loss_op.numpy())
        print(j,'=j, type: target=',type(target),target.shape,'pred',type(pred),pred.shape)
        
        cpuLoss2D=cpuLossF(pred,target).numpy()
        #print(j,'=j, type: cpuLoss2D=',type(cpuLoss2D),cpuLoss2D.shape)
        cpuLossV=np.mean(cpuLoss2D,axis=1)
        #print(j,'=j, type: cpuLossV=',type(cpuLossV),cpuLossV.shape)
        cpuLoss=np.mean(cpuLossV)
        
        print('pred j=%d   ipuLoss=%.4f, cpuLoss=%.4f  Shapes: pred=%s, loss=%s, cpuLossV=%s'%(j,loss,cpuLoss,str(pred.shape),str(loss.shape),str(cpuLossV.shape)))
        '''
        data_dev, target_dev = data.to(device), target.to(device)
        output_dev = model(data_dev)
        lossOp=criterion(output_dev, target_dev)
        print('qq',lossOp,len(test_loader.dataset),len(test_loader)); ok55
        
        output=output_dev.cpu()
        '''
        test_loss += loss
        nEve2=nEve+target.shape[0]
        print('nn',nEve,nEve2)
        Uall[nEve:nEve2,:]=target[:]
        Zall[nEve:nEve2,:]=pred[:]
        nEve=nEve2
        nStep+=1
    test_loss /= nStep
    print('infere done, nEve=%d nStep=%d loss=%.4f'%(nEve,nStep,test_loss))
    return test_loss,Uall,Zall

  
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
  args=get_parser()
  logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

  if args.outPath=='same' : args.outPath=args.modelPath
  sumF=args.modelPath+'/sum_train.yaml'
  sumMD = read_yaml( sumF)
  parMD=sumMD['train_params']
  inpMD=sumMD['input_meta']
  
  model4infer,popOpts=load_model4infer(sumMD,args.modelPath)
  #1print(model)

  if args.cellName!=None:
      parMD['cell_name']=args.cellName

  if args.numSamples!=None:
      parMD['max_samples_per_epoch' ] = args.numSamples
  domain='test'
  parMD['world_size']=1

  data_loader = get_data_loader(parMD,  inpMD,domain, popOpts, verb=args.verb)

  startT=time.time()
  loss,U,Z=model_infer(model4infer,data_loader,sumMD)
  predTime=time.time()-startT
  print('M: infer : Average loss: %.4f  events=%d , elaT=%.2f min\n'% (loss,  Z.shape[0],predTime/60.))

  sumRec={}
  sumRec['domain']=domain
  sumRec[domain+'LossMSE']=float(loss)
  sumRec['predTime']=predTime
  sumRec['numSamples']=U.shape[0]
  sumRec['lossThrHi']=0.50  # for tagging plots
  sumRec['inpShape']=sumMD['train_params']['model']['inputShape']
  sumRec['short_name']=sumMD['train_params']['cell_name']
  sumRec['modelDesign']=sumMD['train_params']['model']['myId']
  sumRec['trainRanks']=sumMD['train_params']['world_size']
  sumRec['trainTime']=sumMD['trainTime_sec']
  sumRec['loss_valid']= sumMD['loss_valid']

  #
  #  - - - -  only plotting code is below - - - - -
  
  plot=Plotter_NeuronInverter(args,inpMD ,sumRec )

  plot.param_residua2D(U,Z)

  write_yaml(sumRec, args.outPath+'/sum_pred.yaml')

  #1plot.params1D(U,'true U',figId=7)
  plot.params1D(Z,'pred Z',figId=8)

  if 0: 
    print('input data example, it will plot waveforms')
    dlit=iter(data_loader)
    xx, yy = next(dlit)
    #1xx, yy = next(dlit) #another sample
    print('batch, X,Y;',xx.shape,xx.dtype,yy.shape,yy.dtype)
    print('Y[:2]',yy[:2])
    plot.frames_vsTime(xx,yy,9)
   
  
  plot.display_all('predict')  

