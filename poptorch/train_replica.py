#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Neuron-Inverter adopet for Graphcore 

Quick test:
 ./train_replica.py --design dev  --epochs 5

For LSTM do:
  ./train_replica.py --design hpar_dev3 (takes longer to start !!!)

Long training on 100 epochs on  10-cell data
  ./train_replica.py --cellName practice10c
  ./train_replica.py --cellName witness2c


Multi-IPU training
m=2
poprun --num-instances=$m --num-replicas=$m   ./train_replica.py --design gc4  --outPath outZ --cellName witness2c

  
'''

import sys,os
from toolbox.Util_IOfunc  import read_yaml, write_yaml
from toolbox.Trainer import Trainer

import argparse
from pprint import pprint

import  logging
import torch
import socket
import popdist

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default='gc4',help='[.hpar.yaml] configuration of model and training')
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-o","--outPath", default='out/', help=' all outputs, also TB')
    parser.add_argument("--cellName", type=str, default='bbp153', help="cell shortName ")
    parser.add_argument("--numInpChan",default=None, type=int, help="if defined, reduces num of input channels")
    parser.add_argument("--initLR",default=None, type=float, help="if defined, replaces learning rate from hpar")
    parser.add_argument("--epochs",default=None, type=int, help="if defined, replaces max_epochs from hpar")
    parser.add_argument("-j","--jobId", default=None, help="optional, aux info to be stored w/ summary")

    args = parser.parse_args()
    return args
  
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == '__main__':

    #..... GC enviroment survey
    host=socket.gethostname()
    device_id = popdist.popdist_core.getDeviceId()
    locReplicas = int(popdist.getNumLocalReplicas())
    total_replicas= int(popdist.getNumTotalReplicas())
    rank = popdist.getInstanceIndex() # index of the current instance 
    world_size = popdist.getNumInstances() # total number of instances
    replicas_per_rank=total_replicas//world_size
    ipus_per_replica=popdist.getNumIpusPerReplica()
    total_ipus=total_replicas*ipus_per_replica
    print("M:I am rank=%d of world=%d on host=%s, locReplias=%d devId %d totReplias=%d totIpus=%d replicas/rank=%d ipu/repl=%d"% (rank, world_size, host,locReplicas,device_id,total_replicas,total_ipus,replicas_per_rank,ipus_per_replica))
    # ....  GC survey done
    
    args = get_parser()
    #for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    params = read_yaml( args.design+'.hpar.yaml',verb=rank==0)
    params['design']=args.design
    params['cell_name']=args.cellName        
    params['num_inp_chan']=args.numInpChan
    params['out_path']=args.outPath   

    # overwrite default configuration
    if args.initLR!=None:
        params['train_conf']['optimizer'][1]= args.initLR
    if args.epochs!=None:
        params['max_epochs']= args.epochs

    # ... rank dependent config .....
    params['world_size'] = world_size
    params['world_rank'] = rank
    params['total_replicas']=total_replicas
   
    params['verb']= args.verb * (rank==0)
    params['job_id']=args.jobId
    
  
    # refine BS for multi-gpu configuration
    tmp_batch_size=params.pop('batch_size')
    if params['const_local_batch']: # faster but LR changes w/ num GPUs
        params['local_batch_size'] =tmp_batch_size
        params['global_batch_size'] =tmp_batch_size*params['total_replicas']
    else:
        params['local_batch_size'] = int(tmp_batch_size//params['world_size'])
        params['global_batch_size'] = tmp_batch_size

    trainer = Trainer(params)
    trainer.train_replica()

    if rank>0: exit(0)
    
    sumF=args.outPath+'/sum_train.yaml'
    write_yaml(trainer.sumRec, sumF) 

    if 0:
        epoch=trainer.sumRec['epoch_stop']
        

    print("M:done design", args.design)
