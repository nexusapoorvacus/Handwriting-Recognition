import sys
from random import shuffle

BASEDIR = "../../images/"

PERCENT_TRAIN = 0.80
PERCENT_VALIDATE = 0.10
PERCENT_TEST = 0.10

if len(sys.argv) < 2:
  print("This function takes in the path to image_path_file.csv")
  sys.exit()

image_path_file = sys.argv[1]
lines = open(image_path_file, "r").readlines()[1:]
shuffled_indices = range(len(lines))
shuffle(shuffled_indices)

train_amt = int(PERCENT_TRAIN * len(shuffled_indices))
validate_amt = int(PERCENT_VALIDATE * len(shuffled_indices))
test_amt = int(PERCENT_TEST * len(shuffled_indices))

train_indices = shuffled_indices[:train_amt]
valid_indices = shuffled_indices[train_amt:train_amt+validate_amt]
test_indices = shuffled_indices[train_amt+validate_amt:train_amt + validate_amt+test_amt]

train_file = open(BASEDIR + "train.lst", "w")
valid_file = open(BASEDIR + "valid.lst", "w")
test_file = open(BASEDIR + "test.lst", "w")

for index in train_indices:
  line = lines[index]
  filename = line[:line.index("|")]
  last_slash_index = filename[::-1].index("/")
  train_file.write(filename[-last_slash_index:] + " " + str(index) + "\n")
for index in valid_indices:
  line = lines[index]
  filename = line[:line.index("|")]
  last_slash_index = filename[::-1].index("/")
  valid_file.write(filename[-last_slash_index:] + " " + str(index) + "\n")
for index in test_indices:
  line = lines[index]
  filename = line[:line.index("|")]
  last_slash_index = filename[::-1].index("/")
  test_file.write(filename[-last_slash_index:] + " " + str(index) + "\n")

train_file.close()
test_file.close()
valid_file.close()
