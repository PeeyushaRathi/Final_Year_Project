import re
from linecache import getline
import pdf2txt
from subprocess import call
import glob
import os
import string 
import sys
for root, dirs, filenames in os.walk('.',topdown=False):
        for name in filenames:
            if(name.endswith('txt')):
                file_name  = os.path.join(root, name)
                print('\n')
                print(file_name)
                f1 = open(file_name,'r')
                data = f1.read()
                regex= [r'(.*?)high(\s*)court',r'(.*?)district(\s*)court']
                for r in regex:    
                    m = re.search(r,data[:200],re.IGNORECASE)
                    if m:
                        print (m.group().strip())
                        break
                    
