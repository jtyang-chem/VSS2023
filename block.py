#!/home/jtyang/anaconda3/bin/python
import pandas as pd
import numpy as np
import time
import os
import re
import fnmatch
# ---
import chStr as js
import jcor
import bfnode as jn

def mk_by_template(fout,fTemplate,subDict):
    """make file by substitude keys in template"""

    # open
    ftemp = open(fTemplate,'r')
    fnew = open(fout,'w')

    # lines
    lines = ftemp.read()

    #for line in lines:
    # substitute
    for key,value in subDict.items():
        lines = lines.replace(key, value)
    fnew.write(lines)
        
    # close
    fnew.close()
    ftemp.close()

def get_info_by_re(fin, reg=re.compile(r'xxx')):
    with open(fin,'r') as f:
        fstr= f.read()
    #res = reg.search( fstr).group(0)
    res = re.findall(reg, fstr)[0]
    return res

def subJob(fin):
    print(os.popen( "qsub "+fin).read())    
    time.sleep(3)
