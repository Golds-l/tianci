import json

from PIL import Image

train_json_re = json.load(open("data/validation.json"))
val_json_re = json.load(open("data/validation.json"))

for i in val_json_re:
    print(i)
    path_img_re = "data/validation_src/{0}".format(i)
    img_re = Image.open(path_img_re)
    if len(val_json_re[i]["label"]) == 1:
        img1_1 = img_re.crop((val_json_re[i]["left"][0], val_json_re[i]["top"][0],
                              val_json_re[i]["left"][0] + val_json_re[i]["width"][0],
                              val_json_re[i]["top"][0] + val_json_re[i]["height"][0]))
        img1_1.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][0], str(val_json_re[i]["label"][0]) + i),
            quality=95,
            subsampling=0)

    if len(val_json_re[i]["label"]) == 2:
        img2_1 = img_re.crop((val_json_re[i]["left"][0], val_json_re[i]["top"][0],
                              val_json_re[i]["left"][0] + val_json_re[i]["width"][0],
                              val_json_re[i]["top"][0] + val_json_re[i]["height"][0]))
        img2_1.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][0], str(val_json_re[i]["label"][0]) + i),
            quality=95,
            subsampling=0)

        img2_2 = img_re.crop((val_json_re[i]["left"][1], val_json_re[i]["top"][1],
                              val_json_re[i]["left"][1] + val_json_re[i]["width"][1],
                              val_json_re[i]["top"][1] + val_json_re[i]["height"][1]))
        img2_2.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][1], str(val_json_re[i]["label"][1]) + i),
            quality=95,
            subsampling=0)

    if len(val_json_re[i]["label"]) == 3:
        img3_1 = img_re.crop((val_json_re[i]["left"][0], val_json_re[i]["top"][0],
                              val_json_re[i]["left"][0] + val_json_re[i]["width"][0],
                              val_json_re[i]["top"][0] + val_json_re[i]["height"][0]))
        img3_1.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][0], str(val_json_re[i]["label"][0]) + i),
            quality=95,
            subsampling=0)

        img3_2 = img_re.crop((val_json_re[i]["left"][1], val_json_re[i]["top"][1],
                              val_json_re[i]["left"][1] + val_json_re[i]["width"][1],
                              val_json_re[i]["top"][1] + val_json_re[i]["height"][1]))
        img3_2.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][1], str(val_json_re[i]["label"][1]) + i),
            quality=95,
            subsampling=0)

        img3_3 = img_re.crop((val_json_re[i]["left"][2], val_json_re[i]["top"][2],
                              val_json_re[i]["left"][2] + val_json_re[i]["width"][2],
                              val_json_re[i]["top"][2] + val_json_re[i]["height"][2]))
        img3_3.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][2], str(val_json_re[i]["label"][2]) + i),
            quality=95,
            subsampling=0)

    if len(val_json_re[i]["label"]) == 4:
        img4_1 = img_re.crop((val_json_re[i]["left"][0], val_json_re[i]["top"][0],
                              val_json_re[i]["left"][0] + val_json_re[i]["width"][0],
                              val_json_re[i]["top"][0] + val_json_re[i]["height"][0]))
        img4_1.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][0], str(val_json_re[i]["label"][0]) + i),
            quality=95,
            subsampling=0)

        img4_2 = img_re.crop((val_json_re[i]["left"][1], val_json_re[i]["top"][1],
                              val_json_re[i]["left"][1] + val_json_re[i]["width"][1],
                              val_json_re[i]["top"][1] + val_json_re[i]["height"][1]))
        img4_2.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][1], str(val_json_re[i]["label"][1]) + i),
            quality=95,
            subsampling=0)

        img4_3 = img_re.crop((val_json_re[i]["left"][2], val_json_re[i]["top"][2],
                              val_json_re[i]["left"][2] + val_json_re[i]["width"][2],
                              val_json_re[i]["top"][2] + val_json_re[i]["height"][2]))
        img4_3.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][2], str(val_json_re[i]["label"][2]) + i),
            quality=95,
            subsampling=0)

        img4_4 = img_re.crop((val_json_re[i]["left"][3], val_json_re[i]["top"][3],
                              val_json_re[i]["left"][3] + val_json_re[i]["width"][3],
                              val_json_re[i]["top"][3] + val_json_re[i]["height"][3]))
        img4_4.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][3], str(val_json_re[i]["label"][3]) + i),
            quality=95,
            subsampling=0)

    if len(val_json_re[i]["label"]) == 5:
        img5_1 = img_re.crop((val_json_re[i]["left"][0], val_json_re[i]["top"][0],
                              val_json_re[i]["left"][0] + val_json_re[i]["width"][0],
                              val_json_re[i]["top"][0] + val_json_re[i]["height"][0]))
        img5_1.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][0], str(val_json_re[i]["label"][0]) + i),
            quality=95,
            subsampling=0)

        img5_2 = img_re.crop((val_json_re[i]["left"][1], val_json_re[i]["top"][1],
                              val_json_re[i]["left"][1] + val_json_re[i]["width"][1],
                              val_json_re[i]["top"][1] + val_json_re[i]["height"][1]))
        img5_2.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][1], str(val_json_re[i]["label"][1]) + i),
            quality=95,
            subsampling=0)

        img5_3 = img_re.crop((val_json_re[i]["left"][2], val_json_re[i]["top"][2],
                              val_json_re[i]["left"][2] + val_json_re[i]["width"][2],
                              val_json_re[i]["top"][2] + val_json_re[i]["height"][2]))
        img5_3.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][2], str(val_json_re[i]["label"][2]) + i),
            quality=95,
            subsampling=0)

        img5_4 = img_re.crop((val_json_re[i]["left"][3], val_json_re[i]["top"][3],
                              val_json_re[i]["left"][3] + val_json_re[i]["width"][3],
                              val_json_re[i]["top"][3] + val_json_re[i]["height"][3]))
        img5_4.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][3], str(val_json_re[i]["label"][3]) + i),
            quality=95,
            subsampling=0)

        img5_5 = img_re.crop((val_json_re[i]["left"][4], val_json_re[i]["top"][4],
                              val_json_re[i]["left"][4] + val_json_re[i]["width"][4],
                              val_json_re[i]["top"][4] + val_json_re[i]["height"][4]))
        img5_5.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][4], str(val_json_re[i]["label"][4]) + i),
            quality=95,
            subsampling=0)

    if len(val_json_re[i]["label"]) == 6:
        img6_1 = img_re.crop((val_json_re[i]["left"][0], val_json_re[i]["top"][0],
                              val_json_re[i]["left"][0] + val_json_re[i]["width"][0],
                              val_json_re[i]["top"][0] + val_json_re[i]["height"][0]))
        img6_1.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][0], str(val_json_re[i]["label"][0]) + i),
            quality=95,
            subsampling=0)

        img6_2 = img_re.crop((val_json_re[i]["left"][1], val_json_re[i]["top"][1],
                              val_json_re[i]["left"][1] + val_json_re[i]["width"][1],
                              val_json_re[i]["top"][1] + val_json_re[i]["height"][1]))
        img6_2.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][1], str(val_json_re[i]["label"][1]) + i),
            quality=95,
            subsampling=0)

        img6_3 = img_re.crop((val_json_re[i]["left"][2], val_json_re[i]["top"][2],
                              val_json_re[i]["left"][2] + val_json_re[i]["width"][2],
                              val_json_re[i]["top"][2] + val_json_re[i]["height"][2]))
        img6_3.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][2], str(val_json_re[i]["label"][2]) + i),
            quality=95,
            subsampling=0)

        img6_4 = img_re.crop((val_json_re[i]["left"][3], val_json_re[i]["top"][3],
                              val_json_re[i]["left"][3] + val_json_re[i]["width"][3],
                              val_json_re[i]["top"][3] + val_json_re[i]["height"][3]))
        img6_4.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][3], str(val_json_re[i]["label"][3]) + i),
            quality=95,
            subsampling=0)

        img6_5 = img_re.crop((val_json_re[i]["left"][4], val_json_re[i]["top"][4],
                              val_json_re[i]["left"][4] + val_json_re[i]["width"][4],
                              val_json_re[i]["top"][4] + val_json_re[i]["height"][4]))
        img6_5.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][4], str(val_json_re[i]["label"][4]) + i),
            quality=95,
            subsampling=0)

        img6_6 = img_re.crop((val_json_re[i]["left"][5], val_json_re[i]["top"][5],
                              val_json_re[i]["left"][5] + val_json_re[i]["width"][5],
                              val_json_re[i]["top"][5] + val_json_re[i]["height"][5]))
        img6_6.save(
            "data/validation/{0}/{1}".format(val_json_re[i]["label"][5], str(val_json_re[i]["label"][5]) + i),
            quality=95,
            subsampling=0)

# val_img_path_src = "data/validation/"
# val_json_path_src = "data/validation.json"
#
#
# def path_process(path_src, json_path):
#     img_path = []
#     img_json = json.load(open(json_path))
#     for i in img_json:
#         path = path_src + i
#         img_path.append(path)
#     return img_path
#
# def label_process(img_json_path):
#     img_json = json.load(open(img_json_path))
#     img_label=[img_json[x]["label"]for x in img_json]
#     return img_label
#
#
# def img_cut(img_path, img_label):
#     for i in range(len(img_label)):
