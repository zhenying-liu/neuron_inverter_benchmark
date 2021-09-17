import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#-------------------
#-------------------
#-------------------
class CNNandFC_Model(nn.Module):
#...!...!..................
    def __init__(self,hpar,verb=0):
        super(CNNandFC_Model, self).__init__()
        if verb: print('CNNandFC_Model hpar=',hpar)
        timeBins,inp_chan=hpar['inputShape']
        self.inp_shape=(inp_chan,timeBins) # swap order
        
        self.verb=verb
        if verb: print('CNNandFC_Model inp_shape=',self.inp_shape,', verb=%d'%(self.verb))
        
        # .....  CNN layersg
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

        ''' Automatically compute the size of the output of CNN+Pool block,  
        needed as input to the first FC layer 
        '''

        with torch.no_grad():
            # process 2 fake examples through the CNN portion of model
            x1=torch.tensor(np.zeros((2,)+self.inp_shape), dtype=torch.float32)
            y1=self.forwardCnnOnly(x1)
            self.flat_dim=np.prod(y1.shape[1:]) 
            if verb>1: print('myNet flat_dim=',self.flat_dim)

        self.flat_bn=None
        if hpar['batch_norm_flat']:
            self.flat_bn=torch.nn.BatchNorm1d(self.flat_dim)
     
        # .... add FC  layers
        hpar2=hpar['fc_block']
        self.fc_block  = nn.ModuleList()
        inp_dim=self.flat_dim
        for i,dim in enumerate(hpar2['dims']):
            self.fc_block.append( nn.Linear(inp_dim,dim))
            inp_dim=dim
            self.fc_block.append( nn.ReLU())
            if hpar2['dropFrac']>0 : self.fc_block.append( nn.Dropout(p= hpar2['dropFrac']))

        #.... the last FC layer may have different activation and no Dropout
        self.fc_block.append(nn.Linear(inp_dim,hpar['outputSize']))
        # here I have chosen no tanh activation so output range is unconstrained
  

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
    def forward(self, x):
        if self.verb>2: print('J: inF',x.shape,'numLayers CNN=',len(self.cnn_block),'FC=',len(self.fc_block))
        x=self.forwardCnnOnly(x)
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

