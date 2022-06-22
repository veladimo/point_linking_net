import os

rootann = r'C:\Users\q1387\Desktop\AI学习\目标检测合集\point-linking-network-master\data\VOCdevkit\VOC2012\Annotations'
roottrainval = r'C:\Users\q1387\Desktop\AI学习\目标检测合集\point-linking-network-master\data\VOCdevkit\VOC2012\ImageSets\Main\\trainval.txt'
roottest = r'C:\Users\q1387\Desktop\AI学习\目标检测合集\point-linking-network-master\data\VOCdevkit\VOC2012\ImageSets\Main\\test.txt'
file_names = os.listdir(rootann)
alls = []
for file_name in file_names:
    file_name = file_name[:-4]
    # print(file_name)
    alls.append(file_name)

with open(roottrainval) as f:
    trainvals = f.readlines()

trainandvals = []
for trainval in trainvals:
    trainandvals.append(trainval[:-1])
# print(trainandvals)

with open(roottest,'w') as f:
    for all in alls:
        if all not in trainandvals:
            f.write(all+'\n')