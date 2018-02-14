import os
import re


for root, dirs, filenames in os.walk('.',topdown=False):
    for name in filenames:
        if(name.endswith('txt')):
            start_index = 0
            end_index = 0
            judgement = ''
            temp = []
            doc = ''
            file_name  = os.path.join(root, name)
            print('\n')
            print(file_name)
            f = (open(file_name, 'r'))
            content = f.read()
            length = len(content)-1400
            start = ['appeal', 'application', 'petition','revision','appellant', 'accused']
            regexs = ['partly allow','partially allow','allowed partly', 'partly',
                      'allow','condon',
                      'dismiss','disposed of', 'reject', 'quash']
            
            for r in start:
                m = re.search(r,content[length:], re.I)
                if m:
                    start_index = m.start(0)
                    doc = m.group().lower()
                    #print(start_index)
                    #print(m.group())
                    break

            for r in regexs:
                m = re.search(r,content[length:], re.I)
                if m:
                    end_index = m.start(0)
                    print(start_index)
                    print(doc)
                    print(end_index)
                    print(m.group())
                    judgement = m.group().lower()
                    break
                            
                    
            if judgement and doc:
                i = regexs.index(judgement)
                j = start.index(doc)
                if j>3:
                    doc = 'Appeal'
                if i<=3:
                    print(doc+" is partly allowed")
                elif 3<i<6:
                    print(doc+" is allowed")
                else:
                    print(doc+" is dismissed")                  
            else:
                print("Not available")       
                     
                
                
                
            
            
               
            
