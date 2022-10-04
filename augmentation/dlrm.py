import random 
seed = 42
random.seed(seed)

input = "/raid/data/dlrm/kaggle/train.txt"
output = f"/raid/data/dlrm/augmentation/kaggle/train{seed}.txt"
f_original = open(input, "r")
f_new = open(output, "w")

count = 1
while True:
    text = f_original.readline()
    if text == "":
            break
    entries = text.split("\t")
    label = entries[:1]
    integer = entries[1:14]
    categorical = entries[14:]
    categorical[-1] = categorical[-1][:-1] # remove the newline character 
    random.shuffle(integer)
    random.shuffle(categorical)
    categorical[-1] = categorical[-1] + '\n' # add the newline character 
    text = "\t".join(label + integer + categorical)
    f_new.write(text)
    count += 1

    if count % 10000 == 0:
        print(count)
    

f_original.close()
f_new.close()


