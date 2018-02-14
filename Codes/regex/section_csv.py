import re
import os

'''
rape_dict = {"376":'IPC', "363":'IPC', "366":'IPC', "323":'IPC', "342":'IPC',
           "506":'IPC', "170":'IPC', "354":'IPC', "419":'IPC', "511":'IPC',
           "324":'IPC', "90":'IPC', "417":'IPC', "504":'IPC', "325":'IPC',
           "313":'IPC', "34":'IPC', "228A":'IPC', "452":'IPC', "377":'IPC',
           "450":'IPC', "362":'IPC', "493":'IPC', "375":'IPC', "368":'IPC',
           "365":'IPC', "344":'IPC', "342":'IPC',"341":'IPC', "420":'IPC',
           "458":'IPC', "354A":'IPC', "354B":'IPC', "354C":'IPC', "392":'IPC',
           "201":'IPC', "374(2)":'Cr.P.C', "156(3)":'Cr.P.C', "378(3)":'Cr.P.C',
           "374":'Cr.P.C', "164":'Cr.P.C', "162":'Cr.P.C', "439":'Cr.P.C',
           "390":'Cr.P.C', "82":'Cr.P.C', "216":'Cr.P.C', "83":'Cr.P.C',
           "357(3)":'Cr.P.C', "482":'Cr.P.C', "35":' Indian Evidence Act',
           "32(5)":'Indian Evidence Act', "114-A":'Indian Evidence Act', "145":'Indian Evidence Act',
           "157":'Indian Evidence Act', "137":'Indian Evidence Act', "4":'POCSO',
           "5(k)":'POCSO', "5(1)(n)":'POCSO', "6":'POCSO', "8":'POCSO',
           "3(3)":'POCSO', "11(iv)":'POCSO', "12":'POCSO', "2(1)(d)":'POCSO',
           "33(7)":'POCSO',"5(d)":'POCSO', "10":'POCSO', "24":'POCSO', "29":'POCSO',
           "26":'POCSO'}

murder_dict = {"302":'IPC', "306":'IPC', "307":'IPC', "498-A":'IPC', "34":'IPC',
               "304B":'IPC', "498A":'IPC', "304":'IPC', "452":'IPC', "449":'IPC',
               "506(11)":'IPC', "448":'IPC', "147":'IPC', "143":'IPC', "148":'IPC',
               "149":'IPC', "37(1)":'IPC', "437-A":'Cr.P.C.', "313":'Cr.P.C.',
               "374(2)":'Cr.P.C.', "512":'Cr.P.C.', "394":'Cr.P.C.', "164":'Cr.P.C.',
               "428":'Cr.P.C.', "161":'Cr.P.C.', "437":'Cr.P.C.', "209":'Cr.P.C.',
               "293":'Cr.P.C.', "172":'Cr.P.C.', "25-A":'Cr.P.C.', "235":'Cr.P.C.',
               "357":'Cr.P.C.', "52":'Transfer of Property Act', "80":'Indian Evidence Act',
               "157":'Indian Evidence Act', "158":'Indian Evidence Act', "8":'Indian Evidence Act',
               "106":'Indian Evidence Act', "32":'Indian Evidence Act', "27":'Indian Evidence Act',
               "135":'Bombay Police Act', "4/25":'Arms Act', "15":'Juvenile Justice Act'}

kidnap_dict = {"326":'IPC', "307":'IPC', "120B":'IPC', "302":'IPC', "448":'IPC',
               "324":'IPC', "364-A":'IPC', "365":'IPC', "201":'IPC', "34":'IPC', "363":'IPC',
               "389":'Cr.P.C.', "374":'Cr.P.C.', "357A":'Cr.P.C.'}

theft_dict = {"420":'IPC', "420/34":'IPC', "406":'IPC', "120B":'IPC', "409":'IPC', "467":'IPC',
              "415":'IPC', "482":'Cr.P.C.', "378(1)(3)":'Cr.P.C.',"235(1)":'Cr.P.C.', "313":'Cr.P.C.',
              "378(3)(1)&(4)":'Cr.P.C.', "245":'Cr.P.C.', "438(2)":'Cr.P.C.', "202":'Cr.P.C.',
              "156(1)":'Cr.P.C.', "155(2)":'Cr.P.C.', "200":'Cr.P.C.', "246":'Cr.P.C.',
              "164":'Cr.P.C.', "281":'Cr.P.C.', "451":'Cr.P.C.', "397":'Cr.P.C.',
              "401":'Cr.P.C.', "464":'Cr.P.C.', "135(1)(A)":'Indian Electricty Act',
              "135":'Indian Electricty Act', "135(1)(B)":'Indian Electricty Act',
              "135(1)":'Indian Electricty Act', "126":'Indian Electricty Act', "127":'Indian Electricty Act',
              "50":'Indian Electricty Act', "154":'Indian Electricty Act', "151":'Indian Electricty Act',
              "39":'Indian Electricty Act', "44":'Indian Electricty Act', "152(2)":'Electricity Supply Code',
              "138":'Negotiable Instruments Act', "139":'Negotiable Instruments Act', "141":'Negotiable Instruments Act',
              "142":'Negotiable Instruments Act'}
'''




import csv
reader = csv.reader(open('sections.csv'))
my_dict = {}
for row in reader:
    key = row[0]
    my_dict[key] = row[1:]

for root, dirs, filenames in os.walk('.',topdown=False):
    for name in filenames:
        if(name.endswith('txt')):
            temp = []
            file_name  = os.path.join(root, name)
            print('\n')
            print(file_name)
            f = open(file_name, 'r')
            content = f.read()
            regex = [ r'Section(.*?\n){2}',
                      r'Section(.*)',
                    ]
            for r in regex:
                m = re.findall(r, content, re.MULTILINE | re.IGNORECASE)
                if m:
                    for line in m:
                        words = re.compile('\w+').findall(line)
                        for w in words:
                            if w in my_dict and w not in temp:
                                  print(w, my_dict[w])
                                  temp.append(w)
