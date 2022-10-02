from inspect import getfile
from pickle import LIST
import subprocess
import os, sys
from typing import List
from Utils import getFileDir

def mine(inputFiles: List[str]):
    cppExecutable = os.path.join(getFileDir(), 'cpp/bin/main')
    processes = []
    for file in inputFiles:
        print(f"Starting IWI-Miner on file: {file}")        
        processes.append(subprocess.Popen([cppExecutable, file]))
    
    
