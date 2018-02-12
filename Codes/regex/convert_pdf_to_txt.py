
import pdf2txt
sample = 'sample.PDF'
from subprocess import call
import glob
for pdf_file in glob.glob('*.PDF'): 
    call(['python.exe','pdf2txt.py', '-o',pdf_file[:-3]+'txt',pdf_file])
#pdf2txt.main([sample, '-o', 'test.txt'])
