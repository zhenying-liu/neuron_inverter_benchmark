#!/usr/bin/env python
'''
Not running on CPUs !

1 or many GPU configuraton is completed in this file

export MASTER_ADDR=`hostname`
srun -n1 python -u train_dist.py 

Runs   1 GPU:  srun -n1 train_dist.py 
or  srun -n2 -l train_dist.py --outPath out2g 

Run on 4 GPUs on 1 node
 salloc -N1 -C gpu  -c 10 --gpus-per-task=1 -t4:00:00  --ntasks-per-node=4 

Full node
 salloc -N1 --ntasks-per-node=8  -C gpu  -c 10   --gpus-per-task=1   --exclusive  -t4:00:00    --image=nersc/pytorch:ngc-21.02-v0


'''

import sys,os
sys.path.append(os.path.abspath("toolbox"))
from Util_IOfunc import read_yaml, write_yaml

import argparse
from pprint import pprint
from Trainer import Trainer 
import  logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
import torch
import torch.distributed as dist

def get_parser():  
  parser = argparse.ArgumentParser()
  parser.add_argument("--design", default='hpar_gcref', help='[.hpar.yaml] configuration of model and training')
  parser.add_argument("-o","--outPath", default='/global/cscratch1/sd/balewski/tmp_digitalMind/neuInv/manual/', type=str)
  parser.add_argument("--facility", default='corigpu', help='data location differes')
  parser.add_argument("--cellName", type=str, default='bbp153', help="cell shortName ")
  parser.add_argument("--numInpChan",default=None, type=int, help="if defined, reduces num of input channels")
  parser.add_argument("--initLR",default=None, type=float, help="if defined, replaces learning rate from hpar")
  parser.add_argument("--epochs",default=None, type=int, help="if defined, replaces max_epochs from hpar")
   

  parser.add_argument("-j","--jobId", default=None, help="optional, aux info to be stored w/ summary")
  parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')

  args = parser.parse_args()
  return args

#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == '__main__':
  args = get_parser()
  if args.verb>2: # extreme debugging
      for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

  os.environ['MASTER_PORT'] = "8879"
  
  params ={}
  #print('M:faci',args.facility)
  if args.facility=='summit':
    import subprocess
    get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ['LSB_DJOB_HOSTFILE'])
    os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    params['local_rank'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
  else:
    #os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    params['local_rank'] = int(os.environ['SLURM_LOCALID'])

  params['world_size'] = int(os.environ['WORLD_SIZE'])
    
  params['world_rank'] = 0
  if params['world_size'] > 1:  # multi-GPU training
    torch.cuda.set_device(params['local_rank'])
    dist.init_process_group(backend='nccl', init_method='env://')
    params['world_rank'] = dist.get_rank()
    #print('M:locRank:',params['local_rank'],'rndSeed=',torch.seed())
  params['verb'] =params['world_rank'] == 0

  #print('M:locRank:',params['local_rank'])
  blob=read_yaml( args.design+'.yaml',verb=params['verb'])
  params.update(blob)
  params['design']=args.design

  if params['verb']:
    logging.info('M: MASTER_ADDR=%s WORLD_SIZE=%s RANK=%s  pytorch:%s'%(os.environ['MASTER_ADDR'] ,os.environ['WORLD_SIZE'], os.environ['RANK'],torch.__version__ ))
    for arg in vars(args):  logging.info('M:arg %s:%s'%(arg, str(getattr(args, arg))))
 
  # refine BS for multi-gpu configuration
  tmp_batch_size=params.pop('batch_size')
  if params['const_local_batch']: # faster but LR changes w/ num GPUs
    params['local_batch_size'] =tmp_batch_size 
    params['global_batch_size'] =tmp_batch_size*params['world_size']
  else:
    params['local_batch_size'] = int(tmp_batch_size//params['world_size'])
    params['global_batch_size'] = tmp_batch_size

  # capture other args values
  params['cell_name']=args.cellName
  params['num_inp_chan']=args.numInpChan
  params['data_path']=params['data_path'][args.facility]
  params['job_id']=args.jobId
  params['out_path']=args.outPath
  # overwrite default configuration
  if args.initLR!=None:
        params['train_conf']['optimizer'][1]= args.initLR
  if args.epochs!=None:
        params['max_epochs']= args.epochs

  trainer = Trainer(params)

  trainer.train()

  if params['world_rank'] == 0:
    sumF=args.outPath+'/sum_train.yaml'
    write_yaml(trainer.sumRec, sumF) # to be able to predict while training continus

    tp=trainer.sumRec['train_params']

    print("M:sum design=%s iniLR=%.1e  epochs=%d  val-loss=%.4f world_size=%d"%(tp['design'],tp['train_conf']['optimizer'][1],tp['max_epochs'],trainer.sumRec['loss_valid'],params['world_size']))
