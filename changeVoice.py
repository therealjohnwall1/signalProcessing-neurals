import os

# preprocessing features 

inner_dir = ["disgust", "happiness", "anger", "fear", "sadness"]
# rename files rm white space
def renameFiles():
    for dir in inner_dir:
        path = "voiceData/" + dir
        print(path)
        count = 0
        for sample in os.listdir(path):
            new_name = f"{dir}_{count}"
            count += 1
            print(path + "/" + sample, path + "/" + new_name)
            os.rename(path + "/" + sample, path +"/" + new_name + ".wav")


def addNoise():




