import os, time, pickle, sys, math,random
import numpy as np
import pandas as pd
import hickle as hkl
import pickle
from datetime import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import straw
chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('hg38_chromosome.txt').readlines()}

!pip install pyBigWig
import pyBigWig
!pip install deeptools
import deeptools
!pip install pysam
import pysam

################################################
#    2. Hi-C preparation (KR normalization)    #
################################################

#1. Define functions 

def prepare_straw(chrom,hic_file,save_name,normalization_method,bin_size=10000):
    '''
    This function uses juicer tools (straw) to extract counts from .hic files
    chrom: which chromosome to extract
    hic_file: location of hic file (e.g. "/content/GM12878.hic")
    save_name: where to save the extracted counts 
    normalization_method: which normalization to use when extracting the counts. 
                          - put "NONE" for extracting raw counts (which will later be used in HiC-DC+)
                          - put "KR" for KR normalization
    bin_size: resolution of extracted Hi-C maps. Default is 10kb (binsize=10000)
    '''
    chrom = str(chrom)
    result = straw.straw(normalization_method,hic_file,chrom,chrom,"BP",bin_size)
    resultdf = pd.DataFrame(np.array(result).T)
    resultdf.to_csv(save_name,sep="\t",header=False,index=False)

def extract_diagonal(chrom,straw_file,save_location,bin_size=10000):
    '''
    This function is used to convert intra-chromosome contact matrices into list of diagonal vectors 
    e.g. input: intra-chromosomal contact matrices constructed from straw files:
         a1 a2 a3 a4 a5 
         a2 b1 b2 b3 b4 
         a3 b2 c1 c2 c3
         a4 b3 c2 d1 d2
         a5 b4 c3 d2 e1 
    output: [[a1,b1,c1,d1,e1],[a2,b2,c2,d2],[a3,b3,c3],[a4,b4],[a5]]

    chrom: chromosome of the straw file 
    straw_file: location of the straw file that extracted from prepare_straw function
    save_location: location for saving the output (list of diagonal values)
    bin_size: resolution of Hi-C contact maps extracted from prepare_straw function
    '''
    # a) construct intra-chromosomal contact map
    mat_dim = int(math.ceil(chrom_len[chrom]*1.0/bin_size)) # create matrix for intra-chromosomal contact map
    contact_matrix = np.zeros((mat_dim,mat_dim))
    for line in open(straw_file).readlines():
        if len(line.strip().split('\t')) == 2:
            idx1, idx2 = int(float(line.strip().split('\t')[0])),int(float(line.strip().split('\t')[1]))
            value = 0.0
        else:
            idx1, idx2, value = int(float(line.strip().split('\t')[0])),int(float(line.strip().split('\t')[1])),float(line.strip().split('\t')[2])
        contact_matrix[int(idx1/bin_size)][int(idx2/bin_size)] = value # count values are filled into the contact map
    # b) extract count values along the diagonals as instructed above
    diag_list = []
    for i in range(1000):
        diag_list.append(np.diagonal(contact_matrix,offset=i).tolist())
    # c) Save into pickle files
    with open(save_location) as fp:
        pickle.dump(diag_list,fp)

#2. Extract counts from .hic files (use juicer straw)

for i in range(1,23):
    prepare_straw(str(i),
                  hic_file="4DNFI1UEG1HD_GM12878.hic",
                  save_name="straw_files/straw_chr"+str(i)+".txt",
                  bin_size=10000)

#2. Extract diagonal vectors

for i in range(1,23):
    extract_diagonal(chrom="chr"+str(i),
                     straw_file="straw_files/straw_chr"+str(i)+".txt",
                     save_location="diagonal/diagonal_chr"+str(i)+".txt",
                     bin_size=10000)