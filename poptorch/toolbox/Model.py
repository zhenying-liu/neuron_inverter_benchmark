__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
#For LSTM results see https://docs.google.com/document/d/1lwqxgStsugrbjM5SDRISv6ei5ILCOlLHF5ecG4dgefs/edit?usp=sharing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import poptorch

from torch.autograd import Variable  #can be differentiated, needed by LSTM
#-------------------
#-------------------
#-------------------
class NeuInvModel(nn.Module):
#...!...!..................
    def __init__(self,params,verb=0):
        super(NeuInvModel, self).__init__()
        
        self.params=params
        hpar = params['model']
        self.verb=verb
        if 'conv_block' in hpar:
            self.add_CNN_block(hpar)
            self.hasCNN=True
        elif 'lstm_block' in hpar:
            self.add_LSTM_block(hpar)
            self.hasCNN=False
        else:
            print('must pick CNN or LSTM block - exiting Model'); exit(98)
              
        self.flat_bn=None
        if hpar['batch_norm_flat']:
            self.flat_bn=torch.nn.BatchNorm1d(self.flat_dim,track_running_stats=False)
               
        # .... add FC  layers
        hpar2=hpar['fc_block']
        self.fc_block  = nn.ModuleList()
        inp_dim=self.flat_dim
        for i,dim in enumerate(hpar2['dims']):
            self.fc_block.append( nn.Linear(inp_dim,dim))
            inp_dim=dim
            self.fc_block.append( nn.ReLU())
            if hpar2['dropFrac']>0 : self.fc_block.append( nn.Dropout(p= hpar2['dropFrac']))

        #.... the last FC layer will have different activation and no Dropout
        self.fc_block.append(nn.Linear(inp_dim,hpar['outputSize']))


#...!...!..................
    def add_CNN_block(self, hpar):
        timeBins,inp_chan=hpar['inputShape']
        self.inp_shape=(inp_chan,timeBins) # swap order
        if self.verb>1 : print('start CNN-block Model inp_shape=',self.inp_shape,', verb=%d'%(self.verb))
        
        # .....  CNN layers
        hpar1=hpar['conv_block']
        self.cnn_block = nn.ModuleList() 
        cnn_stride=1
        for out_chan,cnnker,plker in zip(hpar1['filter'],hpar1['kernel'],hpar1['pool']):
            # class _ConvMd( in_channels, out_channels, kernel_size, stride,
            # CLASS torch.nn.MaxPoolMd(kernel_size, stride=None,                
            self.cnn_block.append( nn.Conv1d(inp_chan, out_chan, cnnker, cnn_stride))
            self.cnn_block.append( nn.MaxPool1d(plker))
            self.cnn_block.append( nn.ReLU())
            inp_chan=out_chan

        # Automatically compute the size of the output of CNN+Pool block,  needed as input to the first FC layer 


        with torch.no_grad():
            # process 2 fake examples through the CNN portion of model
            x1=torch.tensor(np.zeros((2,)+self.inp_shape), dtype=torch.float32)
            y1=self.forwardCnnOnly(x1)
            self.flat_dim=np.prod(y1.shape[1:]) 
            if self.verb>1: print('myNet cnn flat_dim=',self.flat_dim)


#...!...!..................
    def add_LSTM_block(self, hpar):
        timeBins,inp_chan=hpar['inputShape']
        self.inp_shape=(timeBins,inp_chan) # not swap order
        
        if self.verb>1: print('start LSTM-block Model inp_shape=',self.inp_shape,', verb=%d'%(self.verb))

        # .....  LSTM layers
        hpar1=hpar['lstm_block']
        self.num_lstm_layers=hpar1['num_layers']
        
        self.hidden_lstm_size = hpar1['hidden_size']

        self.lstm=nn.LSTM(input_size=inp_chan, hidden_size=self.hidden_lstm_size,num_layers=self.num_lstm_layers, batch_first=True) 

        # test LSTM block
        with torch.no_grad():
            # process 2 fake examples through the LSTM portion of model
            x1=torch.tensor(np.zeros((2,)+self.inp_shape), dtype=torch.float32)
            y1=self.forwardLstmOnly(x1)
            

        self.flat_dim=self.hidden_lstm_size
        if self.verb>1: print('myNet lstm flat_dim=',self.flat_dim)
        
#...!...!..................
    def forwardCnnOnly(self, x):
        # flatten 2D image 
        x=x.view((-1,)+self.inp_shape )

        if self.verb>2: print('J: inp2cnn',x.shape,x.dtype)
        for i,lyr in enumerate(self.cnn_block):
            if self.verb>2: print('Jcnn-lyr: ',i,lyr)
            x=lyr(x)
            if self.verb>2: print('Jcnn: out ',i,x.shape)
        return x
        
#...!...!..................
    def forwardLstmOnly(self, x):

        #x=shape (BS,seq_len,inp_chan)
        if self.verb>2: print('Jlstm: in',x.shape)
        bs=x.size(0)
        h_0 = Variable(torch.zeros(self.num_lstm_layers, bs, self.hidden_lstm_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_lstm_layers, bs, self.hidden_lstm_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_lstm_size) #reshaping the data for Dense layer next
        if self.verb>2: print('Jlstm: hn',hn.shape,h_0.shape)
        return hn
    
  
        
#...!...!..................
    def forward(self, x, target=None):
        if self.verb>2: print('J: inF',x.shape)

        if self.params['gc_m2000']['num_io_tiles'] >= 32:
            x = poptorch.set_overlap_for_input(
                x, poptorch.OverlapMode.OverlapAccumulationLoop)

        if self.hasCNN:
            x=self.forwardCnnOnly(x)
        else:
            x=self.forwardLstmOnly(x)
        x = x.view(-1,self.flat_dim)
        
        if self.flat_bn!=None:
            x=self.flat_bn(x);
            
        for i,lyr in enumerate(self.fc_block):
            x=lyr(x)
            if self.verb>2: print('Jfc: ',i,x.shape)
        if self.verb>2: print('J: y',x.shape)
        return x

#...!...!..................
    def summary(self):
        numLayer=sum(1 for p in self.parameters())
        numParams=sum(p.numel() for p in self.parameters())
        return {'modelWeightCnt':numParams,'trainedLayerCnt':numLayer,'modelClass':self.__class__.__name__}


#-------------------
#-------------------
#-------------------
class MyModelWithLoss(torch.nn.Module): # GC wrapper class
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.MSELoss()

    def forward(self, x,ytrue):
        ypred = self.model(x)
        return ypred, self.loss(ypred,ytrue)

    #Note, forward(.) output can be conditioned on self.training but NOT on ytrue==None (despit the latter may work in some simple cases)
