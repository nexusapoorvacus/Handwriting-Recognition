import sys

if len(sys.argv) < 3:
  print ("This script takes in two arguments. The first is the path to image_path_label.csv. The second is the output directory where the normalized labels will be placed.")
  sys.exit(0)

IMAGE_PATH_LABEL_FILE = sys.argv[1]
DEST_FILE_LOCATION = sys.argv[2]
DEST_FILE_NAME = "labels.norm.lst"

image_path_file = open(IMAGE_PATH_LABEL_FILE, "r")
dest_file = open(DEST_FILE_LOCATION + DEST_FILE_NAME, "w")

for l in image_path_file.readlines()[1:]:
  label = l[l.index("|")+1:]
  label = label.replace(" ", "")
  str_to_write = "|".join([c for c in label])
  dest_file.write(str_to_write)

image_path_file.close()
dest_file.close()

