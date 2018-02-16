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
                f = open(file_name, 'r')
                f1 = open(file_name,'r')
                data = f1.read()
                lines = f.readlines()
                lines[:] = [" ".join(line.split()) for line in lines]
                flag =0
                regexs= [

                        r'for(\s+)the(\s+)appellants?(\s*):(.*?\n){2}',
                        r'for(\s+)appellants?(\s*):(.*?\n){2}',
                        r'for(\s+)petitioners?(\s*):(.*?\n){2}',
                        r'for(\s+)the(\s+)petitioners?(\s*):(.*?\n){2}',
                        r'counsel(\s+)for(\s+)petitioners?(\s*):(.*?\n){2}',
                        r'counsel(\s+)for(\s+)the(\s+)petitioners?(\s*):(.*?\n){2}',
                        r'counsel(\s+)for(\s+)applicants?(\s*):(.*?\n){2}',
                        r'counsel(\s+)for(\s+)the(\s+)applicants?(\s*):(.*?\n){2}',
                        r'represented(\s+)by(\s*):(.*?\n){2}',
                        r'\((\s*)by(.*)\)',
                        r'petitioner(\s+)counsel(\s*):(.*?\n){1}',
                        r'through(\s*):(\s*)(.*?\n){2}',
                        r'appearance(\s*):(\s*)(.*?\n){2}',
                        r'by(\s+)adv(.*?\n){1}',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)appellants?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)appellants?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)petitioners?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)petitioners?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)applicants?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)applicants?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)advocates?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)advs?',
                        
                        ]
                for regex in regexs:
                    m = re.search(regex,data[:3000],re.MULTILINE|re.IGNORECASE)
                    if m:
                        print (m.group().strip())
                        flag =1
                        break

                if flag ==0:
                    regex_lines=[r'for the appellant',r'for appellant',r'counsel for the appellant',r'App for appellant',
                                 r'A.p.p for appellant',r'for the petitioner',r'for petitioner']
                    for regex in regex_lines:
                        for line in lines:
                            if line.lower().lstrip().startswith(regex):
                                ind = lines.index(line)
                                li = lines[ind : ind+2]                
                                print(li)
                                flag = 1
                                break
                            elif line.lower().rstrip().endswith((regex,(regex+'s'))):
                                print(line.strip())
                                flag = 1
                                break
                        if flag==1:
                            break
                        
                if flag ==0:
                    m = re.search(r'present(\s*):(.*?\n){2}',data[:3000],re.MULTILINE|re.IGNORECASE)
                    if m:
                        print (m.group().strip())
                        flag =1
