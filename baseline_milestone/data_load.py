import glob
from string import ascii_lowercase as al
from string import ascii_uppercase as ap

dic = {x:i for i, x in enumerate(al, 0)}
dic2 = {x:i for i, x in enumerate(ap, 26)}
dic.update(dic2)
with open('character_paths.csv', 'w') as f:
    for name in glob.glob('data/by_class/*'):
        sub_dir  = "data/by_class/" + name[14:] + "/train_" + name[14:] + "/*"
        for image_path in glob.glob(sub_dir):
            letter = image_path[14:16].decode("hex")
            ind = dic[letter]
            list_dict = [0]* 52 
            list_dict[ind] = 1
            to_add =  image_path + "," + "|".join([str(s) for s in list_dict]) + "\n"
            f.write(to_add)
