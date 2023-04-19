import os

files = ['ViT-B-16#rebuttal_DA_ADAM_adam#cos-layer4_rebuttal_steps60', 'ViT-B-16#Multi_class_token_Cos_attack_adam#cos-layer4_rebuttal_steps60']

# for i in range(12):
#     files.append('ViT-B-16#Cos_attck_adam#cos-layer{}'.format(i))

for x in files:
    cmd = "python test.py --data_path_name {} --batch_size 16".format(x)
    os.system(cmd)