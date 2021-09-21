__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import time, os
import ruamel.yaml  as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.MantissaNoDotYAML1_1Warning)
import torch # for checkpoints
from pprint import pprint
import csv

#...!...!..................
def read_yaml(ymlFn,verb=1):
        if verb: print('  read  yaml:',ymlFn,end='')
        ymlFd = open(ymlFn, 'r')
        bulk=yaml.load( ymlFd, Loader=yaml.CLoader)
        ymlFd.close()
        xx=os.path.getsize(ymlFn)/1024
        if verb: print(' read yaml:',ymlFn,' size=%.1f kB'%xx)  
        return bulk

#...!...!..................
def write_yaml(rec,ymlFn,verb=1):        
        ymlFd = open(ymlFn, 'w')
        yaml.dump(rec, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close()
        xx=os.path.getsize(ymlFn)/1024
        if verb:
                print('  closed  yaml:',ymlFn,' size=%.1f kB'%xx)

   
#...!...!..................
def read_one_csv(fname,delim=','):
    print('read_one_csv:',fname)
    tabL=[]
    with open(fname) as csvfile:
        drd = csv.DictReader(csvfile, delimiter=delim)
        print('see %d columns'%len(drd.fieldnames),drd.fieldnames)
        for row in drd:
            tabL.append(row)
            
        print('got %d rows \n'%(len(tabL)))
    #print('LAST:',row)
    return tabL,drd.fieldnames

#...!...!..................
def write_one_csv(fname,rowL,colNameL):
    print('write_one_csv:',fname)
    print('export %d columns'%len(colNameL), colNameL)
    with open(fname,'w') as fou:
        dw = csv.DictWriter(fou, fieldnames=colNameL)#, delimiter='\t'
        dw.writeheader()
        for row in rowL:
            dw.writerow(row)    


#...!...!..................
def save_checkpoint( checkpoint_path, model,optimizer,epoch):
    torch.save({ 'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

#...!...!..................
def restore_checkpoint( checkpoint_path,model,optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model=model.load_state_dict(checkpoint['model_state'])
    if optimizer!=None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    startEpoch = checkpoint['epoch'] + 1
    return startEpoch
                            
