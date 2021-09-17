import os,time
from pprint import pprint,pformat
import socket  # for hostname
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import logging

from Model import CNNandFC_Model as MyModel
from Dataloader_H5 import get_data_loader  # 1-cell (code is simpler)
#from Dataloader_multiH5 import get_data_loader  # mutiple cells
from Util_IOfunc import read_yaml

#............................
#............................
#............................
class Trainer():
#...!...!..................
  def __init__(self, params):
    assert torch.cuda.is_available() 
    self.params = params
    self.verb=params['verb']
    self.isRank0=params['world_rank']==0
    self.valPeriod=params['validation_period']
    self.device = torch.cuda.current_device()      
    logging.info('T:ini world rank %d of %d, local rank %d, host=%s  see device=%d'%(params['world_rank'],params['world_size'],params['local_rank'],socket.gethostname(),self.device))

    expDir=params['out_path']
    expDir2=os.path.join(expDir, 'checkpoints/')
    if self.isRank0:
      self.TBSwriter=SummaryWriter(os.path.join(expDir, 'tb_logs'))
      if not os.path.isdir(expDir2):  os.makedirs(expDir2)

    params['checkpoint_name'] =  'checkpoints/ckpt.pth'
    params['checkpoint_path'] = os.path.join(expDir, params['checkpoint_name'])
    params['resuming'] =  params['resume_checkpoint'] and os.path.isfile(params['checkpoint_path'])
    
    optTorch=params['opt_pytorch']
    # EXTRA: enable cuDNN autotuning.
    torch.backends.cudnn.benchmark = optTorch['autotune']

    # AMP: Construct GradScaler for loss scaling
    # AUTOMATIC MIXED PRECISION PACKAGE
    self.grad_scaler = torch.cuda.amp.GradScaler(enabled=optTorch['amp'])

    metaF='%s/meta.cellSpike_%s.yaml'%(params['data_path'],params['probe_type'])
    bulk=read_yaml(metaF,verb=self.verb)
    inpMD=bulk['dataInfo']
    self.inpMD=inpMD

    '''
    try:  # handle practice or witness
      params['numLocalSamples']=params['globalTrainSamples']//params['world_size']  # hack to make 1-cell dataloader to work
      params['cell_name']=inpMD['cellSplit'][ params['cell_name'][0] ]
    except:
      pass
    '''
    
    if self.verb:
      logging.info('T:params %s'%pformat(params))

    # ...... END OF CONFIGURATION .........    
    if self.verb:
      logging.info('T:imported PyTorch ver:%s'%torch.__version__)
      logging.info('T:rank %d of %d, prime data loaders'%(params['world_rank'],params['world_size']))

    params['shuffle']=True
    self.train_loader = get_data_loader(params, inpMD,'train', verb=self.verb)
    '''
    try: 
      params['numLocalSamples']//=8  # reduce smaples for validation & test
    except:
      pass # hack to make 1-cell-dataloader num smaples are taken from data
    '''

    params['shuffle']=True # use False for reproducibility
    params['num_data_workers']*=2 # double # of workers for val-loader
    self.valid_loader = get_data_loader(params,  inpMD,'val', verb=self.verb)
    
    if self.verb:
      logging.info('T:rank %d of %d, data loader initialized'%(params['world_rank'],params['world_size']))
      logging.info('T:train-data: %d steps, localBS=%d, globalBS=%d'%(len(self.train_loader),self.train_loader.batch_size,self.params['global_batch_size']))
      logging.info('T:valid-data: %d steps'%(len(self.valid_loader)))

      if 0:
        print('data example')
        xx, yy = next(iter(self.train_loader))
        print('batch, X,Y;',xx.shape,xx.dtype,yy.shape,yy.dtype)
        print('Y[:10]',yy[:10])
        ok77
    
    # wait for all ranks to finish downloading the data - lets keep some order
    if params['world_size']>1:
      import torch.distributed as dist
      from torch.nn.parallel import DistributedDataParallel
      assert dist.is_initialized()
      dist.barrier()
      self.dist=dist # nneded by this class member methods

    # must know the number of steps to decided how often to print
    self.params['log_freq_step']=max(1,len(self.train_loader)//self.params['log_freq_per_epoch'])

    myModel=MyModel(params['model'], verb=self.verb)

    if self.isRank0 and params['tb_show_graph']:
       dataD=next(iter(self.train_loader))
       images, labels = dataD
       t1=time.time()
       self.TBSwriter.add_graph(myModel,images.to('cpu'))
       t2=time.time()
       if self.verb:  logging.info('show model graph at TB took %.1f sec'%(t2-t1))
    
    self.model=myModel.to(self.device)
    
    if self.verb:
      print('\n\nT: torchsummary.summary(model):');
      from torchsummary import summary
      summary(self.model,(1,4,1600))
      if self.verb>1: print(self.model)

      # save entirel model before training
      modelF ='blank_model.pth'
      params["blank_model"]=modelF
      torch.save( self.model, params['out_path']+'/'+modelF)
      logging.info('T: saved blank model to%s'%modelF)

    tcf=params['train_conf']
    optName, initLR=tcf['optimizer']
    lrcf=tcf['LRsched']
 
    if optTorch['apex']: # EXTRA: use Apex FusedXXX optimizer
      import apex
      if optName=='adam' :
        self.optimizer = apex.optimizers.FusedAdam(self.model.parameters(), lr=initLR) # note, default is adam_w_mode=True
      else:
        print('T:ia invalid Opt:',optName); exit(99)
    else: # NO APEX
      if optName=='adam' :
            self.optimizer=torch.optim.Adam(self.model.parameters(),lr=initLR)
      else:
        print('T:ib invalid Opt:',optName); exit(99)

    
    # choose type of LR decay schedule
    if self.verb: logging.info('LR conf:%s'%str(lrcf))    
    if 'plateau_patience' in lrcf:
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=lrcf['reduceFactor'], patience=lrcf['plateau_patience'], mode='min',cooldown=2, verbose=self.verb)   
    
    self.criterion =torch.nn.MSELoss().to(self.device) # Mean Squared Loss

    if params['world_size']>1:
      self.model = DistributedDataParallel(self.model,
               device_ids=[params['local_rank']],output_device=[params['local_rank']])
      # note, using DDP assures the same as average_gradients(self.model), no need to do it manually ??
    self.iters = 0
    self.startEpoch = 0
    if params['resuming']  and   self.verb:
      logging.info("Loading checkpoint %s"%params['checkpoint_path'])
      self.restore_checkpoint(params['checkpoint_path'])
    self.epoch = self.startEpoch

    if self.verb:  logging.info(self.model)
 
    if self.isRank0:  # create summary record
      self.sumRec={'train_params':params,
                   'host_name' : socket.gethostname(),
                   'num_ranks': params['world_size'],
                   'state': 'model_build',
                   'input_meta':inpMD,
                   'trainTime_sec':-1,
                   'loss_valid':-1,
                   'epoch_start': int(self.startEpoch),
                   'job_id': params['job_id'],
      }

      
#...!...!..................
  def train(self):
    if self.verb:
      logging.info("Starting Training Loop..., myRank=%d resume epoch=%d"%(self.params['world_rank'],self.startEpoch + 1))
    
    bestLoss=1e20  
    startTrain = time.time()
    TperEpoch=[]
    warmup_epochs=self.params['train_conf']['warmup_epochs']
    initLR=self.params['train_conf']['optimizer'][1]

    #. . . . . . .  epoch loop start . . . . . . . . 
    for epoch in range(self.startEpoch, self.params['max_epochs']):
      self.epoch = epoch
      doVal= (epoch %  self.valPeriod[0]) < self.valPeriod[1]
        
      # Apply learning rate warmup     
      if epoch < warmup_epochs:
        self.optimizer.param_groups[0]['lr'] = initLR*float(epoch+1.)/float(warmup_epochs)

      tbeg = time.time()      
      train_logs = self.train_one_epoch()
      t2 = time.time()
      if doVal:  valid_logs = self.validate_one_epoch()
      t3 = time.time()
      tend = time.time()
      
      if epoch >= warmup_epochs and  doVal :
        self.scheduler.step(valid_logs['loss'])
             
      if self.params['save_checkpoint'] or epoch+1==self.params['max_epochs']:
        if self.isRank0 and bestLoss> valid_logs['loss']:
          #checkpoint at the end of every epoch  if loss improved
          self.save_checkpoint(self.params['checkpoint_path'])
          bestLoss= valid_logs['loss']
          logging.info('save_checkpoint for epoch %d , val-loss=%.3g'%(epoch + 1, bestLoss) )

      # . . . .   only logging and histogramming below . . . . .    
      if self.isRank0:
          totT=tend-tbeg
          trainT=t2-tbeg
          valT=t3-t2
          rec1={'train':train_logs['loss']}
          rec2={'train':trainT,'tot':totT,'val':valT}  # time per epoch
          locTotTrainSamp=len(self.train_loader)*self.train_loader.batch_size
          kfac=1000/self.params['world_size']
          rec3={'train':locTotTrainSamp/trainT/kfac}  # train samp/sec
          if epoch>self.startEpoch : TperEpoch.append(totT)
          if doVal:
              rec1['val']=float(valid_logs['loss'])
              locTotValSamp=len(self.valid_loader)*self.valid_loader.batch_size
              rec3.update({'val':float(locTotValSamp/valT/kfac)})  # val samp/sec

          lrTit='LR'
          if self.params['job_id']!=None: lrTit='LR %s'%self.params['job_id']
          self.TBSwriter.add_scalars(' loss ',rec1 , epoch)
          self.TBSwriter.add_scalar(lrTit, self.optimizer.param_groups[0]['lr'], epoch)

          self.TBSwriter.add_scalars('epoch time (sec) ',rec2 , epoch)
          self.TBSwriter.add_scalars('glob_speed (k samp:sec) ',rec3 , epoch)
          tV=np.array(TperEpoch)
          if len(tV)>1:
            tAvr=np.mean(tV); tStd=np.std(tV)/np.sqrt(tV.shape[0])
          else:
            tAvr=tStd=-1

          txt='Epoch %d took %.1f sec, avr=%.2f +/-%.2f sec/epoch, elaT=%.1f sec, nGpu=%d, LR=%.2e, Loss: train=%.4f'%(epoch, totT, tAvr,tStd,time.time() -startTrain,self.params['world_size'] ,self.optimizer.param_groups[0]['lr'],train_logs['loss'])
          if doVal:  txt+=', val=%.4f'%valid_logs['loss']
          if epoch%5==0:
             self.TBSwriter.add_text('summary',txt , epoch)
          if self.verb:  logging.info(txt )

        
    #. . . . . . .  epoch loop end . . . . . . . .
    
    if self.isRank0:  
      self.TBSwriter.add_histogram('time per epoch (sec)', np.array(TperEpoch),epoch)
     
         
      # add info to summary
      try:
        rec={'epoch_stop':epoch+1, 'state':'model_trained','loss_train':float(train_logs['loss']),'loss_valid':float(valid_logs['loss'])}
        rec['trainTime_sec']=time.time()-startTrain
        rec['timePerEpoch_sec']=[float('%.2f'%x) for x in [tAvr,tStd] ]
        self.sumRec.update(rec)
      except:
         if  self.verb:
           logging.warn('trainig  not executed?, summary update failed')

      
#...!...!..................
  def train_one_epoch(self):
    torch.cuda.synchronize()
    report_time = time.time()
    report_bs = 0
    optTorch=self.params['opt_pytorch']
    lossSum = 0.0

    # Loop over training data batches
    for i, data in enumerate(self.train_loader, 0):
      self.iters += 1

      # Move our images and labels to GPU
      images, labels = map(lambda x: x.to(self.device), data)
        
      if optTorch['zerograd']:
        # EXTRA: Use set_to_none option to avoid slow memsets to zero
        self.model.zero_grad(set_to_none=True) # not allowed for torch 1.6?
      else:
        self.model.zero_grad()
      
      self.model.train()
      
      # AMP: Add autocast context manager
      with torch.cuda.amp.autocast(enabled=optTorch['amp']):
        # Model forward pass and loss computation
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

      # AMP: Use GradScaler to scale loss and run backward to produce scaled gradients
      self.grad_scaler.scale(loss).backward()

      # AMP: Run optimizer step through GradScaler (unscales gradients and skips steps if required)
      self.grad_scaler.step(self.optimizer)
      
      # AMP: Update GradScaler loss scale value
      self.grad_scaler.update()

      torch.cuda.synchronize()      
      report_bs += len(images)
      lossSum+=loss.detach()
      
      if i % self.params['log_freq_step'] == 0 and i>0:
        torch.cuda.synchronize()
        if self.verb: logging.info('Epoch: %2d, step: %3d, Avg samp/msec/gpu: %.1f'%(self.epoch, i, report_bs / (time.time() - report_time)/1000.))
        report_time = time.time()
        report_bs = 0

    logs = {'loss': lossSum/len(self.train_loader),}

    if self.params['world_size']>1:
      for key in sorted(logs.keys()):
        logs[key] = torch.as_tensor(logs[key]).to(self.device)
        self.dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/self.dist.get_world_size())

    return logs

  
#...!...!..................
  def validate_one_epoch(self):
    self.model.eval()
    optTorch=self.params['opt_pytorch']
    loss = 0.0

    with torch.no_grad():
      for data in self.valid_loader:
        # Move our images and labels to GPU
        images, labels = map(lambda x: x.to(self.device), data)
        # AMP: Add autocast context manager
        with torch.cuda.amp.autocast(enabled=optTorch['amp']):
          # Model forward pass and loss computation
          outputs = self.model(images)
          loss += self.criterion(outputs, labels)
        
    logs = {'loss': loss/len(self.valid_loader),}

    if self.params['world_size']>1:
      for key in sorted(logs.keys()):
        logs[key] = torch.as_tensor(logs[key]).to(self.device)
        self.dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/self.dist.get_world_size())

    return  logs

#...!...!..................
  def save_checkpoint(self, checkpoint_path, model=None):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """

    if not model:
      model = self.model

    torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

#...!...!..................
  def restore_checkpoint(self, checkpoint_path):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """
    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params['local_rank']))
    self.model.load_state_dict(checkpoint['model_state'])
    self.iters = checkpoint['iters']
    self.startEpoch = checkpoint['epoch'] + 1
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

