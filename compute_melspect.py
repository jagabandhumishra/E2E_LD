import os
import math
import librosa
import numpy as np
import pandas as pd
import argparse
import configparser

class FeatureSave:
  def __init__(self, args, CFG):
    self.args = args
    self.CFG = CFG

  def createpath(self):
    myfiles = os.listdir(self.args.srcdir)
    pathlist = list(map(lambda filename : self.args.srcdir+"/"+filename, myfiles))
    return pathlist

  def writelabel(self):
    df = pd.read_csv(self.args.labelpath, sep='\t', header=None, dtype="str")
    
    pathnlabel = list(map(lambda i: self.args.dstdir+"/my_npyfiles/"+df[0][i]+".npy"+"\t"+df[1][i][:len(np.load(self.args.dstdir+"/my_npyfiles/"+df[0][i]+".npy")+1)], range(len(df))))
    textfile = open(self.args.dstdir+"/Data.txt", "w")
    for element in pathnlabel:
      textfile.write(element + "\n")
    textfile.close()

  def savenpy(self, path, bank):
    direc = self.args.dstdir+"/my_npyfiles"
    if not os.path.exists(direc):
      os.makedirs(direc)
    np.save( direc+"/"+(path.split(".wav")[0]).split("/")[-1] +".npy", bank)
    
  def extractfeatures(self):
    pathlist = self.createpath()
    
    for path in pathlist:
      d,fs = librosa.load(path,sr=None)
      audio = list(map(lambda i : d[i:i+ self.CFG['seg_length']], range(0,len(d), self.CFG['seg_length']) ))[:-1] # remove last segment
      bank = list()
      hann = np.hanning(self.CFG['seg_length'])

      for segment in audio:
        if (self.CFG['need_hann']):
          segment = np.multiply(segment,hann)
        melfb = librosa.feature.melspectrogram(y=segment, n_fft=self.CFG['n_fft'], win_length=self.CFG['win_length'], hop_length=self.CFG['hop_length'],n_mels=self.CFG['n_mels'])
        melfb = np.array(list(map(lambda ele: ele[1:-1], melfb)))
        melfb = np.log10( 1 + melfb.flatten()) # remove first and last feature 21 .> 19
        # melfb = np.log10( 1 + melfb)
        bank.append(melfb)
      self.savenpy(path, bank)
    self.writelabel()      