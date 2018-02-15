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
            length = len(content)-2000
            start_appeal = ['appeal', 'application', 'petition','revision','appellant','accused']
            
            end_result = ['partly allow','partially allow','allowed partly', 'partly',
                          'allow','condon',
                          'dismiss','disposed of','dispose of', 'reject', 'quash']
            
            start_appellant = [r'appellants?(.*?)acquitted',r'accused(.*?)acquitted',
                               r'appellants?(.*?)convicted',r'accused(.*?)convicted',r'accused(.*?)guilty']
            
            for r in start_appeal:
                m = re.search(r,content[length:], re.I)
                if m:
                    doc = m.group().lower()
                    break

            for r in end_result:
                m = re.search(r,content[length:], re.I)
                if m:
                    judgement = m.group().lower()
                    break
                            
            flag = 0        
            if judgement and doc:
                i = end_result.index(judgement)
                if i<=3:
                    print(doc+" is partly allowed")
                elif 3<i<6:
                    print(doc+" is allowed")
                else:
                    print(doc+" is dismissed")                  
            else:
                for s in start_appellant:
                    m = re.search(s,content[length:],re.I|re.DOTALL)
                    if m:
                        judgement1 = m.group().lower()
                        j = start_appellant.index(s)
                        if j<2:
                            print("Appellant is acquitted")
                        else:
                            print("Appellant is convicted")
                        flag = 1
                        break
    
                if flag == 0:
                    print("Not available")
                     
                
                
                
            
            
               
            
