import PIL
from PIL import Image
import sys

if len(sys.argv) < 3:
  print("This script takes in the path to the image_path_label.csv file and the directory with your image data")
  sys.exit(0)

CSV_FILE = sys.argv[1]
DATA_DIR = sys.argv[2]

#csv_file = open("/data/apoorvad/cs221proj/words/data/image_path_label.csv", "r").readlines()
#DATA_DIR = "/data/apoorvad/cs221proj/words/"

for i, lines in enumerate(open(CSV_FILE, "r").readlines()):
  if i == 0:
    continue
  path = DATA_DIR + lines[:lines.index("|")]
  img = Image.open(path)
  img = img.resize((120, 50), PIL.Image.BILINEAR)
  img.save(path)
  if i % 1000 == 0:
    print i
  

