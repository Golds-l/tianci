import json
from PIL import Image
import matplotlib.pyplot as plt

json_path = "data/validation.json"
img_path_src = "data/validation_src"
target_path = "data/validation/10"
json_src = json.load(open(json_path))

for i in json_src:
    img_path = "{0}/{1}".format(img_path_src, i)
    img_src = Image.open(img_path)
    img_10_1 = img_src.crop((0, 0, 32, 64))
    img_10_1.save("{0}/1{1}".format(target_path, i))
    print("{0}/1{1}".format(target_path, i))
    # (left+width/2-32,top,left+width/2,top+64)
    img_height = json_src[i]["height"][0]
    img_width = json_src[i]["width"][0]
    img_top = json_src[i]["top"][0]
    img_left = json_src[i]["left"][0]
    if img_height > 64 and img_width > 32:
        print("ps")
        img_10_2 = img_src.crop((img_left + img_width / 2 - 32, img_top, img_left + img_width / 2, img_top + 64))
        img_10_2.save("{0}/2{1}".format(target_path, i))
        img_10_3 = img_src.crop((img_left, img_top + img_height / 2 - 64, img_left + 32, img_top + img_height / 2))
        img_10_2.save("{0}/3{1}".format(target_path, i))
