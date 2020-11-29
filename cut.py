import json

from PIL import Image

json_src = json.load(open("data/validation.json"))
img_path_src = "data/validation_src"
img_path_tar = "data/validation"

for i in json_src:
    print(i)
    path_img_re = "{1}/{0}".format(i, img_path_src)
    img_re = Image.open(path_img_re)
    if len(json_src[i]["label"]) == 1:
        img1_1 = img_re.crop((json_src[i]["left"][0], json_src[i]["top"][0],
                              json_src[i]["left"][0] + json_src[i]["width"][0],
                              json_src[i]["top"][0] + json_src[i]["height"][0]))
        img1_1.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][0], str(json_src[i]["label"][0]) + i, img_path_tar),
            quality=95,
            subsampling=0)

    if len(json_src[i]["label"]) == 2:
        img2_1 = img_re.crop((json_src[i]["left"][0], json_src[i]["top"][0],
                              json_src[i]["left"][0] + json_src[i]["width"][0],
                              json_src[i]["top"][0] + json_src[i]["height"][0]))
        img2_1.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][0], str(json_src[i]["label"][0]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img2_2 = img_re.crop((json_src[i]["left"][1], json_src[i]["top"][1],
                              json_src[i]["left"][1] + json_src[i]["width"][1],
                              json_src[i]["top"][1] + json_src[i]["height"][1]))
        img2_2.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][1], str(json_src[i]["label"][1]) + i, img_path_tar),
            quality=95,
            subsampling=0)

    if len(json_src[i]["label"]) == 3:
        img3_1 = img_re.crop((json_src[i]["left"][0], json_src[i]["top"][0],
                              json_src[i]["left"][0] + json_src[i]["width"][0],
                              json_src[i]["top"][0] + json_src[i]["height"][0]))
        img3_1.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][0], str(json_src[i]["label"][0]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img3_2 = img_re.crop((json_src[i]["left"][1], json_src[i]["top"][1],
                              json_src[i]["left"][1] + json_src[i]["width"][1],
                              json_src[i]["top"][1] + json_src[i]["height"][1]))
        img3_2.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][1], str(json_src[i]["label"][1]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img3_3 = img_re.crop((json_src[i]["left"][2], json_src[i]["top"][2],
                              json_src[i]["left"][2] + json_src[i]["width"][2],
                              json_src[i]["top"][2] + json_src[i]["height"][2]))
        img3_3.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][2], str(json_src[i]["label"][2]) + i, img_path_tar),
            quality=95,
            subsampling=0)

    if len(json_src[i]["label"]) == 4:
        img4_1 = img_re.crop((json_src[i]["left"][0], json_src[i]["top"][0],
                              json_src[i]["left"][0] + json_src[i]["width"][0],
                              json_src[i]["top"][0] + json_src[i]["height"][0]))
        img4_1.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][0], str(json_src[i]["label"][0]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img4_2 = img_re.crop((json_src[i]["left"][1], json_src[i]["top"][1],
                              json_src[i]["left"][1] + json_src[i]["width"][1],
                              json_src[i]["top"][1] + json_src[i]["height"][1]))
        img4_2.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][1], str(json_src[i]["label"][1]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img4_3 = img_re.crop((json_src[i]["left"][2], json_src[i]["top"][2],
                              json_src[i]["left"][2] + json_src[i]["width"][2],
                              json_src[i]["top"][2] + json_src[i]["height"][2]))
        img4_3.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][2], str(json_src[i]["label"][2]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img4_4 = img_re.crop((json_src[i]["left"][3], json_src[i]["top"][3],
                              json_src[i]["left"][3] + json_src[i]["width"][3],
                              json_src[i]["top"][3] + json_src[i]["height"][3]))
        img4_4.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][3], str(json_src[i]["label"][3]) + i, img_path_tar),
            quality=95,
            subsampling=0)

    if len(json_src[i]["label"]) == 5:
        img5_1 = img_re.crop((json_src[i]["left"][0], json_src[i]["top"][0],
                              json_src[i]["left"][0] + json_src[i]["width"][0],
                              json_src[i]["top"][0] + json_src[i]["height"][0]))
        img5_1.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][0], str(json_src[i]["label"][0]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img5_2 = img_re.crop((json_src[i]["left"][1], json_src[i]["top"][1],
                              json_src[i]["left"][1] + json_src[i]["width"][1],
                              json_src[i]["top"][1] + json_src[i]["height"][1]))
        img5_2.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][1], str(json_src[i]["label"][1]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img5_3 = img_re.crop((json_src[i]["left"][2], json_src[i]["top"][2],
                              json_src[i]["left"][2] + json_src[i]["width"][2],
                              json_src[i]["top"][2] + json_src[i]["height"][2]))
        img5_3.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][2], str(json_src[i]["label"][2]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img5_4 = img_re.crop((json_src[i]["left"][3], json_src[i]["top"][3],
                              json_src[i]["left"][3] + json_src[i]["width"][3],
                              json_src[i]["top"][3] + json_src[i]["height"][3]))
        img5_4.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][3], str(json_src[i]["label"][3]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img5_5 = img_re.crop((json_src[i]["left"][4], json_src[i]["top"][4],
                              json_src[i]["left"][4] + json_src[i]["width"][4],
                              json_src[i]["top"][4] + json_src[i]["height"][4]))
        img5_5.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][4], str(json_src[i]["label"][4]) + i, img_path_tar),
            quality=95,
            subsampling=0)

    if len(json_src[i]["label"]) == 6:
        img6_1 = img_re.crop((json_src[i]["left"][0], json_src[i]["top"][0],
                              json_src[i]["left"][0] + json_src[i]["width"][0],
                              json_src[i]["top"][0] + json_src[i]["height"][0]))
        img6_1.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][0], str(json_src[i]["label"][0]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img6_2 = img_re.crop((json_src[i]["left"][1], json_src[i]["top"][1],
                              json_src[i]["left"][1] + json_src[i]["width"][1],
                              json_src[i]["top"][1] + json_src[i]["height"][1]))
        img6_2.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][1], str(json_src[i]["label"][1]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img6_3 = img_re.crop((json_src[i]["left"][2], json_src[i]["top"][2],
                              json_src[i]["left"][2] + json_src[i]["width"][2],
                              json_src[i]["top"][2] + json_src[i]["height"][2]))
        img6_3.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][2], str(json_src[i]["label"][2]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img6_4 = img_re.crop((json_src[i]["left"][3], json_src[i]["top"][3],
                              json_src[i]["left"][3] + json_src[i]["width"][3],
                              json_src[i]["top"][3] + json_src[i]["height"][3]))
        img6_4.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][3], str(json_src[i]["label"][3]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img6_5 = img_re.crop((json_src[i]["left"][4], json_src[i]["top"][4],
                              json_src[i]["left"][4] + json_src[i]["width"][4],
                              json_src[i]["top"][4] + json_src[i]["height"][4]))
        img6_5.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][4], str(json_src[i]["label"][4]) + i, img_path_tar),
            quality=95,
            subsampling=0)

        img6_6 = img_re.crop((json_src[i]["left"][5], json_src[i]["top"][5],
                              json_src[i]["left"][5] + json_src[i]["width"][5],
                              json_src[i]["top"][5] + json_src[i]["height"][5]))
        img6_6.save(
            "{2}/{0}/{1}".format(json_src[i]["label"][5], str(json_src[i]["label"][5]) + i, img_path_tar),
            quality=95,
            subsampling=0)

#
#
# def path_process(pat,img_path_tarh_src, json_path):
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
