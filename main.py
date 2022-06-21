import os
import math
import librosa
import numpy as np
import argparse
import configparser

from scipy.ndimage.fourier import fourier_shift
from compute_melspect import FeatureSave

def get_configurations(path):
  config = configparser.ConfigParser()
  config.read(path)
  section = config['MAIN']
  CFG = {
      'fs': int(section['fs']),
      'n_mels': int(section['n_mels']),
      'n_fft': int(section['n_fft']),
      'seg_length': int(section['seg_length']),
      'hop_length': int(float(section['hop_len'])*int(section['fs'])),
      'win_length': int(float(section['win_len'])*int(section['fs'])),
      'need_hann': section.getboolean('need_hann')
  }
  return CFG

def __init__():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_path', type=str, required=True, help='path to config.ini')
  parser.add_argument('--srcdir', type=str, required=True, help='path to source directory')
  parser.add_argument('--dstdir', type=str, required=True, help='path to destination directory')
  parser.add_argument('--labelpath', type=str, required=True, help='path to labels')
  args = parser.parse_args()
  return args

if __name__=='__main__':
  args = __init__()
  CFG = get_configurations(args.config_path)
  FeatureSave(args, CFG).extractfeatures()

#!python3 main.py --config_path "/content/drive/MyDrive/IIT_DH/MyCodes/config.ini" --srcdir "/content/drive/MyDrive/IIT_DH/temp/src" --dstdir "/content/drive/MyDrive/IIT_DH/temp"