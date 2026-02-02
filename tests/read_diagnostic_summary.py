
import codecs
import sys
try:
    with codecs.open(r'c:\Trading\Projects\AlphaGPT\tests\artifacts\diagnose_v2_output.txt', 'r', 'utf-16') as f:
        lines = f.readlines()
except:
    with codecs.open(r'c:\Trading\Projects\AlphaGPT\tests\artifacts\diagnose_v2_output.txt', 'r', 'utf-8') as f:
        lines = f.readlines()
    
summary_start = -1
for i, line in enumerate(lines):
    if "SUMMARY" in line:
        summary_start = i
        break

if summary_start != -1:
    for line in lines[summary_start:]:
        print(line.strip())
else:
    # Print last 50 lines anyway
    for line in lines[-50:]:
        print(line.strip())
