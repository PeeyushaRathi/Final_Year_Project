import re
import os
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
