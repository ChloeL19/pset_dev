#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:40:17 2018

@author: chloeloughridge
"""

# a file for dealing with tensorboard

# for running tensorboard
import subprocess, re, os

# naming the next run in tensorboard --> keeps visualization clear
def name_run(dataset):
    # taking a look in the directory that stores previous model runs
    # in order to come up with a good name for the new model run
    n = 0
    if os.path.isdir("logs"):
        # list the file names in that directory
        command = 'cd ./logs; ls'
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        proc_stdout = str(process.communicate()[0].strip())
        # figure out how many times the particular dataset is listed
        pattern = "_" + dataset
        n = len(re.findall(pattern, proc_stdout))
    # return a name for the new run that is unique by giving it the next higher
    # version number   
    return dataset + "_{}".format(n+1)

