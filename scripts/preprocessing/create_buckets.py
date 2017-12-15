import numpy as np
import re
from PIL import Image
import random
from tqdm import tqdm
import sys

VOCAB_FILE = "../../images/vocab.txt"
LABELS_LIST = "../../images/labels.norm.lst"
BUCKETS_LST_DIR = "../../images/"

if len(sys.argv) < 2:
  print("This script takes in either test, valid or train as an argument")
  sys.exit()

vocab = open(VOCAB_FILE).readlines()
formulae = open(LABELS_LIST,'r').readlines()
char_to_idx = {x.split('\n')[0]:i for i,x in enumerate(vocab)}
# print len(char_to_idx)
char_to_idx['#UNK'] = len(char_to_idx)
char_to_idx['#START'] = len(char_to_idx)
char_to_idx['#END'] = len(char_to_idx)
idx_to_char = {y:x for x,y in char_to_idx.iteritems()}
# print char_to_idx['#UNK']
# print char_to_idx['#START']
# print char_to_idx['#END']
print len(char_to_idx)

set = sys.argv[1] # Use train, valid or test to generate corresponding files
file_list = open(BUCKETS_LST_DIR+set+".lst",'r').readlines()
set_list = []
missing = {}
for i,line in enumerate(file_list):
    if int(line.split()[1]) >= len(formulae):
      import pdb; pdb.set_trace()
    form = formulae[int(line.split()[1])].strip().split("|")
    if form[-1] == "":
      form = form[:-1]
    out_form = [char_to_idx['#START']]
    for c in form:
        try:
            out_form += [char_to_idx[c]]
        except:
            if c not in missing:
                print c, " not found!"
                missing[c] = 1
            else:
                missing[c] += 1
            import pdb; pdb.set_trace()
            out_form += [char_to_idx['#UNK']]
    out_form += [char_to_idx['#END']]
    set_list.append([line.split()[0],out_form])
    
buckets = {}
import os
file_not_found_count = 0
for x,y in tqdm(set_list):
    if os.path.exists('./processed/'+x): 
        img_shp = Image.open('./processed/'+x).size
        try:
            buckets[img_shp] += [(x,y)]
        except:
            buckets[img_shp] = [(x,y)]
    else:
        #print x
        file_not_found_count += 1

print "Num files found in %s set: %d/%d"%(set,len(set_list)-file_not_found_count,len(set_list))

print missing
print "Max length of sequence: ",max([len(x[1]) for x in set_list])
## test
print "Testing!"
print buckets[random.choice(buckets.keys())][0]
np.save(set+'_buckets',buckets)

properties = {}
properties['vocab_size'] = len(vocab)
properties['vocab'] = vocab
properties['char_to_idx'] = char_to_idx
properties['idx_to_char'] = idx_to_char
import numpy as np
np.save('properties',properties)
