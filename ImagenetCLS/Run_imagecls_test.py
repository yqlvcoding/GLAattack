import os

files = [
        'ViT-B-16#Multi_class_token_Cos_attack_adam#cos-layer4_rebuttal_steps60',
        'ViT-B-16#rebuttal_DA_ADAM_adam#cos-layer4_rebuttal_steps60'
        ]

# for x in files:
#     cmd = "python test.py --data-path-name {}".format(x)
#     os.system(cmd)


for x in files:
    cmd = "python test.py --data-path-name {}".format(x)
    os.system(cmd)