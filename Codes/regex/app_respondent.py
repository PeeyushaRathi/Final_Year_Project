<<<<<<< HEAD
import re
from linecache import getline
import pdf2txt
from subprocess import call
import glob
import os
import string 
import sys
print('advocates')
for root, dirs, filenames in os.walk('.',topdown=False):
        print(root)
        print('\n')
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
                        
                        r'for(\s+)the(\s+)respondents?(\s*):(.*?\n){2}',
                        r'for(\s+)respondents?(\s*):(.*?\n){2}',
                        r'for(\s+)opposite(\s+)party(\s*):(.*?\n){2}',
                        r'for(\s+)the(\s+)opposite(\s+)party(\s*):(.*?\n){2}',
                        r'for(\s+)the(\s+)state(\s*):(.*?\n){1}',
                        r'counsel(\s+)for(\s+)respondents?(\s*):(.*?\n){2}',
                        r'counsel(\s+)for(\s+)the(\s+)respondents?(\s*):(.*?\n){2}',
                        r'represented(\s+)by(\s*):(.*?\n){2}',
                        r'\((\s*)by(.*)\)',
                        r'respondent(\s+)counsel(\s*):(.*?\n){1}',
                        r'through(\s*):(\s*)(.*?\n){2}',
                        r'appearance(\s*):(\s*)(.*?\n){2}',
                        r'by(\s+)adv.(\s+)public(\s+)prosecutor(.*?\n){1}',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)respondents?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)respondents?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)states?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)states?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)opponents?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)opponents?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)app ',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)a.p.p',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)addl',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)public(\s+)prosecutor',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)additional ',
                        
                        ]
                for regex in regexs:
                    m = re.search(regex,data[:3000],re.MULTILINE|re.IGNORECASE)
                    if m:
                        print (m.group().strip())
                        flag =1
                        break

                if flag ==0:
                    regex_lines=[r'for the respondent',r'for respondent',r'for the state',r'for the opposite party',
                                 r'for respondent/state',r'for the respondent/state']
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

                




'''
for regex in regex_lines:
    for line in lines:
        n = re.findall(regex, line, re.MULTILINE|re.IGNORECASE)

        r'for the appellant (.*?\n){2}',
        r'for appellant(.*?\n){2}',r'for petitioner(.*?\n){2}'

'''
=======
import re
from linecache import getline
import pdf2txt
from subprocess import call
import glob
import os
import string 
import sys
print('advocates')
for root, dirs, filenames in os.walk('.',topdown=False):
        print(root)
        print('\n')
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
                        
                        r'for(\s+)the(\s+)respondents?(\s*):(.*?\n){2}',
                        r'for(\s+)respondents?(\s*):(.*?\n){2}',
                        r'for(\s+)opposite(\s+)party(\s*):(.*?\n){2}',
                        r'for(\s+)the(\s+)opposite(\s+)party(\s*):(.*?\n){2}',
                        r'for(\s+)the(\s+)state(\s*):(.*?\n){1}',
                        r'counsel(\s+)for(\s+)respondents?(\s*):(.*?\n){2}',
                        r'counsel(\s+)for(\s+)the(\s+)respondents?(\s*):(.*?\n){2}',
                        r'represented(\s+)by(\s*):(.*?\n){2}',
                        r'\((\s*)by(.*)\)',
                        r'respondent(\s+)counsel(\s*):(.*?\n){1}',
                        r'through(\s*):(\s*)(.*?\n){2}',
                        r'appearance(\s*):(\s*)(.*?\n){2}',
                        r'by(\s+)adv.(\s+)public(\s+)prosecutor(.*?\n){1}',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)respondents?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)respondents?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)states?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)states?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)opponents?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)for(\s+)the(\s+)opponents?',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)app ',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)a.p.p',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)addl',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)public(\s+)prosecutor',
                        r'(mrs|mr|ms|sri|shri|smt|dr)(.*?)(?:\n*|\r\n*)(.*?)additional ',
                        
                        ]
                for regex in regexs:
                    m = re.search(regex,data[:3000],re.MULTILINE|re.IGNORECASE)
                    if m:
                        print (m.group().strip())
                        flag =1
                        break

                if flag ==0:
                    regex_lines=[r'for the respondent',r'for respondent',r'for the state',r'for the opposite party',
                                 r'for respondent/state',r'for the respondent/state']
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

                




'''
for regex in regex_lines:
    for line in lines:
        n = re.findall(regex, line, re.MULTILINE|re.IGNORECASE)

        r'for the appellant (.*?\n){2}',
        r'for appellant(.*?\n){2}',r'for petitioner(.*?\n){2}'

'''
>>>>>>> Peeyusha
