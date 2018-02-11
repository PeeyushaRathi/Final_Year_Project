
import os
import re
#path = 'doc/'
with open('output2.txt') as f:
    content = f.read()
    #pattern = re.compile('Bench: \s*([^.]+|\S+)')
    #print (pattern.findall(content))
    #regexp2 = re.compile('cria(.*)')
    #print (regexp2.findall(content)[0])
    '''For judge'''
    regexs = [re.compile('Author:(.*)'),re.compile('HON\'BLE(.*)')]
    for r in regexs:
        m = r.findall(content)
        if m:
            print (m)
            break
        else:
            print("No match")
    
    
    '''regexp = re.compile('Bench:(.*)')
    print (regexp.findall(content)[0])
    regexp1 = re.compile('Shri(.*)')
    #print (regexp1.findall(content)[0])
    #print (regexp1.findall(content)[1])
    court = re.compile('(.*)High')
    #print (court.findall(content)[0])
    section = re.compile('Section \w+ \w+ \w+')
    print(section.findall(content)[0])
    print(section.findall(content)[1])
    print(section.findall(content)[2])


    adv = re.findall(r'Mr.(.*?)Advocate', content,re.DOTALL)
    print(adv[0])
    '''
