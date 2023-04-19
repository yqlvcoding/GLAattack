import numpy as np

with open("correct.txt", "r") as fp:
    b = np.loadtxt(fp, dtype=np.int32, delimiter=',')

print(len(b))

with open("data/msvd_data/test_list.txt", 'r') as fp:
    video_ids = [itm.strip() for itm in fp.readlines()]

data = np.array(video_ids)
indics = np.array(b)
with open("data/msvd_data/correct_test_list.txt", 'w') as fp:
    np.savetxt(fp, data[indics], fmt='%s', delimiter=',')



