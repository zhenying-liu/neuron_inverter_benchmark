__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import copy # to create copy of options
import ctypes
import os
import time
from pprint import pprint,pformat
import socket  # for hostname
import numpy as np
import numpy.distutils
import torch
from torch.utils.tensorboard import SummaryWriter
import popart
import poptorch
import popdist
import popdist.poptorch

import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

from toolbox.Model import NeuInvModel , MyModelWithLoss
from toolbox.Dataloader_h5 import get_data_loader
from toolbox.Util_IOfunc import read_yaml, save_checkpoint

import pickle
pickle.DEFAULT_PROTOCOL=4

import libpvti as pvti
channel = pvti.createTraceChannel("LBL")

#............................
#............................
#............................
class Trainer():
#...!...!..................

  def _get_poptorch_options(self, params, for_training):

    popOpts = popdist.poptorch.Options()
    popOpts.deviceIterations(params['gc_m2000']['replica_steps_per_iter']) # Device "step"

    #if for_training:
    #  popOpts.Training.gradientAccumulation(params['gc_m2000']['gradientAccumulation'])
    #else:
    #  popOpts.Training.gradientAccumulation(1)


    if 'num_io_tiles' in params['gc_m2000'] and params['gc_m2000']['num_io_tiles'] >= 32:
      print("using io tiles")
      popOpts.TensorLocations.numIOTiles(params['gc_m2000']['num_io_tiles'])
      popOpts.setExecutionStrategy(poptorch.ShardedExecution())
    #popOpts.anchorMode(poptorch.AnchorMode.All)
    popOpts.outputMode(poptorch.OutputMode.All)

    if self.params['fp16_model']:
      popOpts.Precision.setPartialsType(torch.half)
    if params['gc_m2000']['enableSyntheticData']:
      popOpts.enableSyntheticData(True)
    if params['gc_m2000']['graph_caching']:
      cachePath='./exec_cache'
      popOpts.enableExecutableCaching(cachePath)
      if  self.verb: logging.info('caching to %s'%(cachePath))

    if 'prefetch_depth' in self.params['gc_m2000']:
      popOpts._Popart.set("defaultPrefetchBufferingDepth", self.params['gc_m2000']['prefetch_depth'])

    if self.isDist:
      popOpts.randomSeed(42+ params['world_rank']) # force the different Droput sequence on each IPU

    return popOpts


  def __init__(self, params):

    self.params = params
    self.verb=params['verb']

    self.isRank0=params['world_rank']==0
    self.valPeriod=params['validation_period']
    self.isDist=params['world_size']>1
    self.compiled = False
    self.validation = params['validation']

    self.device = popdist.popdist_core.getDeviceId()
    logging.info('T:ini world rank %d of %d, host=%s  see device=%s'%(params['world_rank'],params['world_size'],socket.gethostname(),str(self.device)))

    expDir=params['out_path']
    if self.isRank0:
        self.TBSwriter=SummaryWriter(os.path.join(expDir, 'tb_logs'))
        expDir2=os.path.join(expDir, 'checkpoints')
        if not os.path.isdir(expDir2):  os.makedirs(expDir2)


    #params['checkpoint_path'] = os.path.join(expDir, 'checkpoints/ckpt.pth')
    params['checkpoint_path'] = os.path.join(expDir, 'ckpt.pth')
    params['resuming'] =  params['resume_checkpoint'] and os.path.isfile(params['checkpoint_path'])


    if self.verb:
        logging.info('T:params %s'%pformat(params))

    # ...... END OF CONFIGURATION .........
    if self.verb:
      logging.info('imported PyTorch:%s  PopTorch:%s'%(torch.__version__,poptorch.__version__))
      logging.info('rank %d, begin data loader init'%params['world_rank'])

    metaF='%s/meta.cellSpike_%s.yaml'%(params['data_path'],params['probe_type'])
    bulk=read_yaml(metaF,verb=self.verb)
    inpMD=bulk['dataInfo']
    self.inpMD=inpMD

    if self.isDist:
      import horovod.torch as hvd
      hvd.init()
      self.hvd=hvd
      if self.verb: logging.info('T:horovod started, num ranks=%d, stagger_delay %d sec/rank'%(hvd.size(),params['gc_m2000']['stagger_delay_sec']))

      # it may be an overkill, but w-load of 500 can't be healthy, mostlikely it is due to IO from up to 16 HD5 from data loaders and/or  rading cached graphs
      delayMe=params['gc_m2000']['stagger_delay_sec']* params['world_rank']
      time.sleep(delayMe)

    popOpts = self._get_poptorch_options(params, for_training=True)

    if 'use_all_reduce' in self.params['gc_m2000'] and self.params['gc_m2000']['use_all_reduce']:
      so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_ops.so")
      if os.path.exists(so_path):
        ctypes.cdll.LoadLibrary(so_path)
      else:
        logging.info("Could not find custom_ops.so. Execute `make` before running this script.")
      logging.info("All reduce custom op in : {}".format(so_path))

    #popOpts._Popart.setPatterns({"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True, "ZFusedReplicatedAllReducePattern": True})

    self.train_loader = get_data_loader(params, inpMD,'train', popOpts,verb=self.verb)
    if self.validation:
        if self.valPeriod[1]>0:
            self.valid_loader = get_data_loader(params,  inpMD,'val', popOpts, verb=self.verb)
            if self.params['gc_m2000']['pseudoValidation']: next(iter(self.valid_loader)) # HACK, otherwise  training loop will stuck on 1st val-pass
            if self.verb: logging.info('valid-data: %d steps, localBS*repStep*repli=%d'%(len(self.valid_loader),self.valid_loader.batch_size))

    if self.verb:
      logging.info('rank %d of %d, data loader initialized, valPeriod=%s'%(params['world_rank'],params['world_size'],str(self.valPeriod)))
      logging.info('train-data: %d steps, localBS*replicaStep=%d, globalBS=%d'%(len(self.train_loader),self.train_loader.batch_size,self.params['global_batch_size']))


    if 0:
        print('\ntrain data example')
        xx, yy = next(iter(self.train_loader))
        print('train batch, X,Y;',xx.shape,xx.dtype,yy.shape,yy.dtype)
        print('Y[:2]',yy[:2])
        print('\ntrain data example')
        xx, yy = next(iter(self.train_loader))
        print('train batch, X,Y;',xx.shape,xx.dtype,yy.shape,yy.dtype)
        print('Y[:2]',yy[:2])
        ok77


    # must know the number of steps to decided how often to print
    self.params['log_freq_step']=max(1,len(self.train_loader)//self.params['log_freq_per_epoch'])


    myModel=NeuInvModel(params['model'], verb=self.verb)

    if self.params['fp16_model']:
      myModel = myModel.half()

    modelWloss=MyModelWithLoss(myModel)
    if self.isDist:
      hvd.broadcast_parameters(modelWloss.state_dict(), root_rank=0)

    if self.verb>1:
      print('\n\nT: torchsummary.summary(myModel):')
      print(myModel)
      #from torchsummary import summary
      #from torchinfo import summary
      #summary(myModel,(1,4,1600))#, batch_size=1, device='cuda')

      # save entirel model before training
      modelF = params['out_path']+'/blank_model.pth'
      torch.save(myModel, modelF)
      print('T: saved blank model',modelF)
      params["blank_path"]=modelF


    tcf=params['train_conf']
    lrcf=tcf['LRsched']

    if self.verb: logging.info('optimizer:%s'%str(tcf['optimizer']))
    optName, initLR=tcf['optimizer']
    if optName=='AdamW':
      self.optimizer = poptorch.optim.AdamW(myModel.parameters(), lr=initLR)
      self.fakeOptimizer = poptorch.optim.AdamW(myModel.parameters(), lr=0., betas=(0.999,0.999), weight_decay=0) # it will not modify weights

    else:
      print('unknown optimizer %s, abort'%optName); exit(99)

    avrAttrL=['lr','betas','weight_decay']
    for x in avrAttrL:
        self.optimizer.variable_attrs.markAsVariable(x)
        self.fakeOptimizer.variable_attrs.markAsVariable(x)

    if self.verb:
      for x in avrAttrL:
        logging.info('optimizer attr:%s isConst:%r'%(x,self.optimizer.variable_attrs.isConstant("x")))


    if self.verb: logging.info("Poptorch create model start ...")
    self.model4train = poptorch.trainingModel(modelWloss, options=popOpts, optimizer=self.optimizer)
    if  self.valPeriod[1]>0:
        inferencePopOpts = self._get_poptorch_options(params, for_training=False)
        self.model4infer = poptorch.inferenceModel(modelWloss, options=inferencePopOpts)
        if self.verb: logging.info("Poptorch create inference model done")

    # choose type of LR decay schedule
    if self.verb: logging.info('LR conf:%s'%str(lrcf))
    self.doRedLRplat=False
    if 'plateau_patience' in lrcf:
        assert self.valPeriod[1]>0 # needs validation loss
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=lrcf['reduceFactor'], patience=lrcf['plateau_patience'], mode='min',cooldown=2, verbose=self.verb)
        self.doRedLRplat=True

    if 'decay_epochs' in lrcf:
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lrcf['decay_epochs'], gamma=lrcf['gamma'], verbose=self.verb)



    self.iters = 0
    self.startEpoch = 0
    if params['resuming']  and self.params['world_rank'] == 0:
      logging.info("Loading checkpoint %s"%params['checkpoint_path'])
      self.restore_checkpoint(params['checkpoint_path'])
    self.epoch = self.startEpoch

    if self.verb>1:   #debug-level
      logging.info(myModel)

    if self.isRank0:  # create summary record
      self.sumRec={'train_params':params,
                   'hostName' : socket.gethostname(),
                   'numRanks': params['world_size'],
                   'state': 'model_build',
                   'input_meta':inpMD,
                   'trainTime_sec':-1,
                   'loss_valid':-1,
                   'epoch_start': int(self.startEpoch),
                   'job_id': params['job_id'],
      }


#...!...!..................
  def train_replica(self):
    if self.verb:
      logging.info("Starting Training Loop..., myRank=%d resume epoch=%d"%(self.params['world_rank'],self.startEpoch + 1))

    bestLoss=1e20
    TperEpoch=[]
    warmup_epochs=self.params['train_conf']['warmup_epochs']
    optName, initLR=self.params['train_conf']['optimizer']

    pvti.Tracepoint.begin(channel, "train_replica")

    startTrain = time.time()
    #. . . . . . .  epoch loop start . . . . . . . .
    for epoch in range(self.startEpoch, self.params['max_epochs']):

      self.doVal = False
      self.epoch= epoch
      # decide if validation runs
      if self.validation:
          self.doVal= (epoch %  self.valPeriod[0]) < self.valPeriod[1]
          if self.valPeriod[1]>0 and epoch >= self.params['max_epochs']-2:
              if self.verb: logging.info('use true val-pass for epoch=%d'%epoch)
              self.doVal=True
              self.params['gc_m2000']['pseudoValidation']=False

      # Apply learning rate warmup for some optimizers
      if epoch < warmup_epochs:
          self.optimizer.param_groups[0]['lr'] = initLR*float(epoch+1.)/float(warmup_epochs)
          self.model4train.setOptimizer(self.optimizer) # propagate LR to compiled grap

      t1 = time.time()
      train_logs = self.train_one_epoch(self.train_loader)
      t2 = time.time()

      if self.validation:
          if self.doVal :
              if self.params['gc_m2000']['pseudoValidation']:
                  # use Alex trick: no graph swapping but switch to pseudo-training using optimizer w/ LR=0
                  self.model4train.setOptimizer(self.fakeOptimizer) # AdamW w/ LR-0
                  t3 = time.time()
                  valid_logs = self.train_one_epoch(self.valid_loader)
                  t4 = time.time()
                  self.model4train.setOptimizer(self.optimizer) # restore training
                  t5 = time.time()
              else:
                  #.... do graph swap WORKING - very slow
                  self.model4train.detachFromDevice() #GC needs it
                  if self.model4infer._executable:  self.model4infer.attachToDevice()
                  t3 = time.time()
                  valid_logs = self.validate_one_epoch()
                  t4 = time.time()
                  self.model4infer.detachFromDevice() #GC needs it
                  if self.model4train._executable:  self.model4train.attachToDevice()
                  t5 = time.time()
              loss_val=np.mean(valid_logs['loss'])

      tend = time.time()
      loss_train=np.mean(train_logs['loss'])


      if epoch >= warmup_epochs and  self.doVal :
        if self.doRedLRplat:
          self.scheduler.step(loss_val)
        else:
          self.scheduler.step()
        self.model4train.setOptimizer(self.optimizer) #GC:  propagate LR to compiled graph

      if self.isRank0:
          totT=tend-t1
          trainT=t2-t1
          rec1={'train': loss_train}
          rec2={'train':trainT,'tot':totT,'val':0.,'gr-swap':0.}  # time per epoch
          locTotTrainSamp=len(self.train_loader)*self.train_loader.batch_size
          kfac=1000/self.params['world_size']
          rec3={'train':locTotTrainSamp/trainT/kfac}  # train samp/sec

          if self.doVal:
              valT=t4-t3
              swapT=t5-t4 + t3-t2
              rec1['val']=loss_val
              rec2.update({'val':valT,'gr-swap':swapT})
              locTotValSamp=len(self.valid_loader)*self.valid_loader.batch_size
              rec3.update({'val':float(locTotValSamp/valT/kfac)})  # val samp/sec

          lrTit='LR'
          if self.params['job_id']!=None: lrTit='LR %s'%self.params['job_id']
          self.TBSwriter.add_scalars('loss',rec1 , self.epoch)
          self.TBSwriter.add_scalar(lrTit, self.optimizer.param_groups[0]['lr'], self.epoch)

          self.TBSwriter.add_scalars('epoch time (sec) ',rec2 , self.epoch)
          self.TBSwriter.add_scalars('glob_speed (k samp:sec) ',rec3 , self.epoch)
          if epoch>self.startEpoch  and self.params['gc_m2000']['pseudoValidation']: TperEpoch.append(totT)  # use only mid-stream
          tV=np.array(TperEpoch)
          if len(tV)>0:
            tAvr=np.mean(tV); tStd=np.std(tV)/np.sqrt(tV.shape[0])
          else:
            tAvr=tStd=0

          # txt='Epoch %d took %.3f sec, avr=%.2f +/-%.2f sec/epoc, elaT=%.1f sec, nIPU=%d, LR=%.2e, Loss: train=%.4f'%(
          #   epoch, totT, tAvr,tStd,time.time() -startTrain,self.params['total_replicas'] ,self.optimizer.param_groups[0]['lr'],loss_train
          # )

          txt='Epoch %d took %.3f sec (train + DL + logging) avg %.3f sec, avg step time %.3f sec, epoch time with DL %.3f sec, tput %.2f samp/sec, nIPU=%d, LR=%.2e, Loss: train=%.4f'%(
            epoch, totT, tAvr, train_logs['step_time'], train_logs['epoch_time'], train_logs['tput'], self.params['total_replicas'] ,self.optimizer.param_groups[0]['lr'],loss_train
          )
          if self.doVal:
            pseu='pseudo-' if self.params['gc_m2000']['pseudoValidation'] else ''
            txt+=', %sval=%.4f'%(pseu,loss_val)
          if epoch%5==0:
              self.TBSwriter.add_text('summary',txt , epoch)
          if self.verb:  logging.info(txt)

      if self.isRank0:
        if self.params['save_checkpoint'] and bestLoss> valid_logs['loss']:
          testMe_90
          #checkpoint at the end of every epoch  if loss improved
          self.save_checkpoint(self.params['checkpoint_path'])
          bestLoss= valid_logs['loss']
          logging.info('save_checkpoint for epoch %d , val-loss=%.3g'%(epoch + 1, bestLoss) )

    #. . . . . . .  epoch loop end . . . . . . . .

    if self.params['world_rank'] == 0:  # create summary record
      outF2=os.path.join(self.params['out_path'],self.params['checkpoint_path'])
      save_checkpoint(outF2,self.model4train,self.optimizer,epoch)
      logging.info('E:training saved:%s'%outF2)
      # add info to summary
      try:
        rec={'epoch_stop':epoch+1, 'state':'model_trained','loss_train':float(loss_train)}
        rec['trainTime_sec']=time.time()-startTrain
        if self.doVal: rec['loss_valid']=float(loss_val)
        self.sumRec.update(rec)
      except:
         if self.params['log_to_screen'] and self.verb:
           logging.warn('trainig  not executed?')
    pvti.Tracepoint.end(channel, "train_replica")

#...!...!..................
  def train_one_epoch(self,dataLoader):

    self.model4train.train()
    step_time = []
    report_bs = 0
    t1 = time.time()
    # Graphcore speciffic
    loss=0
    for ist, (data, target) in enumerate(dataLoader):
        if not self.compiled:
          self.model4train.compile(data, target)
          self.compiled = True

        loss = 1
        report_time = time.time() # reset timer
        _, loss_op = self.model4train(data, target)
        train_time = time.time() - report_time
        loss += loss_op.numpy()

        step_time.append(train_time)
        report_bs += data.size()[0]
        # print('x'*60, ' ' ,data.size(),report_bs )
        # only reports speed in mid-epoch
        if ist % self.params['log_freq_step'] == 0:
          if self.verb: logging.info('Epoch: %2d, train step: %3d, train time: %.3f sec, Avg samp/sec/instance: %.1fK'%(self.epoch, ist, train_time, 1e-3*data.size()[0] / (time.time() - report_time)))
          # report_time = time.time()
          # report_bs = 0
    t2 = time.time()
    loss /= len(dataLoader)
    if self.isDist>0:
        loss = np.mean(self.hvd.allgather_object(loss))

    logs = {'loss': loss, 'step_time': np.mean(step_time), 'epoch_time': t2 - t1, 'tput': report_bs/(t2-t1)}
    return logs


#...!...!..................
  def validate_one_epoch(self):
    self.model4infer.eval()
    loss=0
    for i, (data, target) in enumerate(self.valid_loader):
        _, loss_op = self.model4infer(data, target)
        loss += loss_op.numpy()

    loss /= len(self.valid_loader)
    if self.isDist>0:
        loss = np.mean(self.hvd.allgather_object(loss))

    logs = {'loss': loss,}
    return logs


#...!...!..................
  def save_checkpoint(self, checkpoint_path, model=None):
    not_working_GC
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
