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

#############################################
#    1. ChIP-seq preparation for GM12878    #
#############################################

#1. DNaseI
#replicate 1
!wget https://www.encodeproject.org/files/ENCFF020WZB/@@download/ENCFF020WZB.bam
#replicate 2
!wget https://www.encodeproject.org/files/ENCFF729UYK/@@download/ENCFF729UYK.bam
# Merge two replicates
file_name = "GM12878_DNaseI"
pysam.merge("/content/"+str(file_name)+"_merge.bam","/content/ENCFF020WZB.bam","/content/ENCFF729UYK.bam")
pysam.index("/content/"+str(file_name)+"_merge.bam")
# bamCoverage
!bamCoverage --bam /content/GM12878_DNaseI_merge.bam -o /content/drive/MyDrive/Research/Predict2Dv2/data/bigWig/GRCh38/GM12878_DNaseI_merge.bigWig --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398 --ignoreForNormalization chrX --extendReads 200

#2. CTCF
#replicate 1
!wget https://www.encodeproject.org/files/ENCFF430XCG/@@download/ENCFF430XCG.bam
#replicate 2
!wget https://www.encodeproject.org/files/ENCFF794BPW/@@download/ENCFF794BPW.bam
# Merge two replicates
file_name = "GM12878_CTCF"
pysam.merge("/content/"+str(file_name)+"_merge.bam","/content/ENCFF430XCG.bam","/content/ENCFF794BPW.bam")
pysam.index("/content/"+str(file_name)+"_merge.bam")
# bamCoverage
!bamCoverage --bam /content/GM12878_CTCF_merge.bam -o /content/drive/MyDrive/Research/Predict2Dv2/data/bigWig/GRCh38/GM12878_CTCF_merge.bigWig --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398 --ignoreForNormalization chrX --extendReads 200

#3. H3K4me3
#replicate 1
!wget https://www.encodeproject.org/files/ENCFF938DOA/@@download/ENCFF938DOA.bam
#replicate 2
!wget https://www.encodeproject.org/files/ENCFF540GUY/@@download/ENCFF540GUY.bam
# Merge two replicates
file_name = "GM12878_H3K4me3"
pysam.merge("/content/"+str(file_name)+"_merge.bam","/content/ENCFF938DOA.bam","/content/ENCFF540GUY.bam")
pysam.index("/content/"+str(file_name)+"_merge.bam")
# bamCoverage
!bamCoverage --bam /content/GM12878_H3K4me3_merge.bam -o /content/drive/MyDrive/Research/Predict2Dv2/data/bigWig/GRCh38/GM12878_H3K4me3_merge.bigWig --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398 --ignoreForNormalization chrX --extendReads 200

#4. H3K27me3
#replicate 1
!wget https://www.encodeproject.org/files/ENCFF265UBT/@@download/ENCFF265UBT.bam
#replicate 2
!wget https://www.encodeproject.org/files/ENCFF824VSE/@@download/ENCFF824VSE.bam
# Merge two replicates
file_name = "GM12878_H3K27me3"
pysam.merge("/content/"+str(file_name)+"_merge.bam","/content/ENCFF265UBT.bam","/content/ENCFF824VSE.bam")
pysam.index("/content/"+str(file_name)+"_merge.bam")
# bamCoverage
!bamCoverage --bam /content/GM12878_H3K27me3_merge.bam -o /content/drive/MyDrive/Research/Predict2Dv2/data/bigWig/GRCh38/GM12878_H3K27me3_merge.bigWig --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398 --ignoreForNormalization chrX --extendReads 200

#5. H3K27ac
#replicate 1
!wget https://www.encodeproject.org/files/ENCFF269GKF/@@download/ENCFF269GKF.bam
#replicate 2
!wget https://www.encodeproject.org/files/ENCFF201OHW/@@download/ENCFF201OHW.bam
# Merge two replicates
file_name = "GM12878_H3K27ac"
pysam.merge("/content/"+str(file_name)+"_merge.bam","/content/ENCFF269GKF.bam","/content/ENCFF201OHW.bam")
pysam.index("/content/"+str(file_name)+"_merge.bam")
# bamCoverage
!bamCoverage --bam /content/GM12878_H3K27ac_merge.bam -o /content/drive/MyDrive/Research/Predict2Dv2/data/bigWig/GRCh38/GM12878_H3K27ac_merge.bigWig --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398 --ignoreForNormalization chrX --extendReads 200

#6. Cohesin (SMC3)
# downloaded from: https://www.encodeproject.org/experiments/ENCSR000DZP/https://www.encodeproject.org/experiments/ENCSR000DZP/
#replicate 1
!wget https://www.encodeproject.org/files/ENCFF302PYC/@@download/ENCFF302PYC.bam
#replicate 2
!wget https://www.encodeproject.org/files/ENCFF622ERY/@@download/ENCFF622ERY.bam
# Merge two replicates
file_name = "GM12878_SMC3"
pysam.merge("/content/"+str(file_name)+"_merge.bam","/content/ENCFF302PYC.bam","/content/ENCFF622ERY.bam")
pysam.index("/content/"+str(file_name)+"_merge.bam")
# bamCoverage
!bamCoverage --bam /content/GM12878_SMC3_merge.bam -o /content/drive/MyDrive/Research/Predict2Dv2/data/bigWig/GRCh38/GM12878_SMC3_merge.bigWig --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2913022398 --ignoreForNormalization chrX --extendReads 200

