import sys

def read_sentences_text_file(sentence_file_path):
  max_width = 0
  max_height = 0
  with open(sentence_file_path) as infile:
    for line in infile:
      # ignore commented out lines
      if line[0] == "#":
        continue
      # parse file path from file name
      # format: xxx-xxxx-xxx-xx or xxx-xxx-xxx-xx
      # example: a01-000u-s00-00
      width = int(line.split(" ")[5])
      height = int(line.split(" ")[6])
      max_width = max(max_width, width)
      max_height = max(max_height, height)
      folder_one = line[:line.index("-")] # a01
      rest = line[line.index("-") + 1:]
      folder_two = folder_one + "-" + rest[:rest.index("-")] #a01-000u
      rest = rest[rest.index("-") + 1:]
      part_three = rest[:rest.index("-")] #s00
      rest = rest[rest.index("-") + 1:]
      part_four = rest[:2] # 00
      image_name = folder_two + "-" + part_three + "-" + part_four + ".png"
      image_path = "data" + "/" + folder_one + "/" + folder_two + "/" + image_name
      label = line.split(" ")[-1].replace("|", " ")[:-1]
      csv_file.write(image_path + "|" + label + "\n")
  print "max_width: " + str(max_width)
  print "max_height: " + str(max_height)

# MAIN
if len(sys.argv) != 2:
  print("Specify the location of sentences.txt - default location is in data/")
  sys.exit(0)
sentences_file_location = sys.argv[1]
csv_file = open('image_path_label.csv', 'w')
csv_file.write('Path|Label\n')
read_sentences_text_file(sentences_file_location)
