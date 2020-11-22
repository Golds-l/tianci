import glob
import cv2
import json
from PIL import Image
import matplotlib.pylab as plt

path = glob.glob("input/train/*.png")
json_re = json.load(open("input/train.json"))

num = 30000
dic = {}
for i in json_re:
    path_img_re = "input/train/{0}".format(i)
    img_re = Image.open(path_img_re)

    print("input/train/{0}".format(i))

    if len(json_re[i]["label"]) == 2:
        # print(img_name)
        # print(json_re[i]["top"][0], json_re[i]["height"][0], json_re[i]["left"][0], json_re[i]["width"][0])
        img2_1 = img_re.crop((json_re[i]["left"][0], json_re[i]["top"][0],
                              json_re[i]["left"][0] + json_re[i]["width"][0],
                              json_re[i]["top"][0] + json_re[i]["height"][0]))
        img2_2 = img_re.crop((json_re[i]["left"][1], json_re[i]["top"][1],
                              json_re[i]["left"][1] + json_re[i]["width"][1],
                              json_re[i]["top"][1] + json_re[i]["height"][1]))

        img2_n = img_re

        img2_n.paste(img2_2, (json_re[i]["left"][0], json_re[i]["top"][0]))
        img2_n.paste(img2_1, (json_re[i]["left"][0] + json_re[i]["width"][1], json_re[i]["top"][0]))
        img_name = "0{0}.png".format(num)

        img2_n.save("input/train_augmentation/{0}".format(img_name), quality=95, subsampling=0)
        img2_label = [0, 0]
        img2_label[0] = json_re[i]["label"][1]
        img2_label[1] = json_re[i]["label"][0]
        dic2_2 = {"label": img2_label}
        dic2_1 = {"{0}".format(img_name): dic2_2}
        dic.update(dic2_1)
        num += 1

    if len(json_re[i]["label"]) == 3:
        img3_1 = img_re.crop((json_re[i]["left"][0], json_re[i]["top"][0],
                              json_re[i]["left"][0] + json_re[i]["width"][0],
                              json_re[i]["top"][0] + json_re[i]["height"][0]))
        img3_2 = img_re.crop((json_re[i]["left"][1], json_re[i]["top"][1],
                              json_re[i]["left"][1] + json_re[i]["width"][1],
                              json_re[i]["top"][1] + json_re[i]["height"][1]))
        img3_3 = img_re.crop((json_re[i]["left"][2], json_re[i]["top"][2],
                              json_re[i]["left"][2] + json_re[i]["width"][2],
                              json_re[i]["top"][2] + json_re[i]["height"][2]))
        # 六种组合方式 img_re = 123
        # 1- 132
        img_re = Image.open(path_img_re)
        img3_n1 = img_re
        img3_n1.paste(img3_3, (json_re[i]["left"][0] + json_re[i]["width"][0], json_re[i]["top"][1]))
        img3_n1.paste(img3_2, (json_re[i]["left"][0] + json_re[i]["width"][0] + json_re[i]["width"][2],
                               json_re[i]["top"][2]))
        img_name = "0{0}.png".format(num)
        img3_n1.save("input/train_augmentation/{0}".format(img_name), quality=95, subsampling=0)
        img3_n1_label = [0, 0, 0]
        img3_n1_label[0] = json_re[i]["label"][0]
        img3_n1_label[1] = json_re[i]["label"][2]
        img3_n1_label[2] = json_re[i]["label"][1]
        dict3_n1_2 = {"label": img3_n1_label}
        dict3_n1_1 = {"{0}".format(img_name): dict3_n1_2}
        dic.update(dict3_n1_1)
        num += 1

        # 2- 213
        img_re = Image.open(path_img_re)
        img3_n2 = img_re
        img3_n2.paste(img3_2, (json_re[i]["left"][0], json_re[i]["top"][0]))
        img3_n2.paste(img3_1, (json_re[i]["left"][0] + json_re[i]["width"][1], json_re[i]["top"][1]))
        img3_n2.paste(img3_3, (json_re[i]["left"][0] + json_re[i]["width"][1] + json_re[i]["width"][0],
                               json_re[i]["top"][2]))
        img_name = "0{0}.png".format(num)
        img3_n2.save("input/train_augmentation/{0}".format(img_name), quality=95, subsampling=0)
        img3_n2_label = [0, 0, 0]
        img3_n2_label[0] = json_re[i]["label"][1]
        img3_n2_label[1] = json_re[i]["label"][0]
        img3_n2_label[2] = json_re[i]["label"][2]
        dict3_n2_2 = {"label": img3_n2_label}
        dict3_n2_1 = {"{0}".format(img_name): dict3_n2_2}
        dic.update(dict3_n2_1)
        num += 1

        # 3- 231
        img_re = Image.open(path_img_re)
        img3_n3 = img_re
        img3_n3.paste(img3_2, (json_re[i]["left"][0], json_re[i]["top"][0]))
        img3_n3.paste(img3_3, (json_re[i]["left"][0] + json_re[i]["width"][1], json_re[i]["top"][1]))
        img3_n3.paste(img3_1, (json_re[i]["left"][0] + json_re[i]["width"][1] + json_re[i]["width"][2],
                               json_re[i]["top"][2]))
        img_name = "0{0}.png".format(num)
        img3_n3.save("input/train_augmentation/{0}".format(img_name), quality=95, subsampling=0)
        img3_n3_label = [0, 0, 0]
        img3_n3_label[0] = json_re[i]["label"][1]
        img3_n3_label[1] = json_re[i]["label"][2]
        img3_n3_label[2] = json_re[i]["label"][0]
        dict3_n3_2 = {"label": img3_n3_label}
        dict3_n3_1 = {"{0}".format(img_name): dict3_n3_2}
        dic.update(dict3_n3_1)
        num += 1

        # 4- 312
        img_re = Image.open(path_img_re)
        img3_n4 = img_re
        img3_n4.paste(img3_3, (json_re[i]["left"][0], json_re[i]["top"][0]))
        img3_n4.paste(img3_1, (json_re[i]["left"][0] + json_re[i]["width"][2], json_re[i]["top"][1]))
        img3_n4.paste(img3_2, (json_re[i]["left"][0] + json_re[i]["width"][2] + json_re[i]["width"][0],
                               json_re[i]["top"][2]))
        img_name = "0{0}.png".format(num)
        img3_n4.save("input/train_augmentation/{0}".format(img_name), quality=95, subsampling=0)
        img3_n4_label = [0, 0, 0]
        img3_n4_label[0] = json_re[i]["label"][2]
        img3_n4_label[1] = json_re[i]["label"][0]
        img3_n4_label[2] = json_re[i]["label"][1]
        dict3_n4_2 = {"label": img3_n4_label}
        dict3_n4_1 = {"{0}".format(img_name): dict3_n4_2}
        dic.update(dict3_n4_1)
        num += 1

        # 5- 321
        img_re = Image.open(path_img_re)
        img3_n5 = img_re
        img3_n5.paste(img3_3, (json_re[i]["left"][0], json_re[i]["top"][0]))
        img3_n5.paste(img3_2, (json_re[i]["left"][0] + json_re[i]["width"][2], json_re[i]["top"][1]))
        img3_n5.paste(img3_1, (json_re[i]["left"][0] + json_re[i]["width"][2] + json_re[i]["width"][1],
                               json_re[i]["top"][2]))
        img_name = "0{0}.png".format(num)
        img3_n5.save("input/train_augmentation/{0}".format(img_name), quality=95, subsampling=0)
        img3_n5_label = [0, 0, 0]
        img3_n5_label[0] = json_re[i]["label"][2]
        img3_n5_label[1] = json_re[i]["label"][1]
        img3_n5_label[2] = json_re[i]["label"][0]
        dict3_n5_2 = {"label": img3_n5_label}
        dict3_n5_1 = {"{0}".format(img_name): dict3_n5_2}
        dic.update(dict3_n5_1)
        num += 1


json_up = json_re
json_up.update(dic)
with open("input/augmentation.json", "w") as fp:
    fp.write(json.dumps(json_up))
