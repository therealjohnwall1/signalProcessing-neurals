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



# move data into new directories
def trainSplit(trainSplit):
    train,test, val = trainSplit
    assert(round(train + test + val)== 1)

    # training dataset first
    for dir in inner_dir:
        path = "voiceData/" + dir
        files = os.listdir(path)
        train_size = int(len(files) * train)
        
        train_files = files[:train_size]
        test_size = int(len(files) * test)
        test_files = files[train_size:train_size + test_size]
        val_files = files[train_size + test_size:]
        
        train_dir = "train/" + dir
        test_dir = "test/" + dir
        val_dir = "val/" + dir
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        for file in train_files:
            os.rename(os.path.join(path, file), os.path.join(train_dir, file))
        
        for file in test_files:
            os.rename(os.path.join(path, file), os.path.join(test_dir, file))

        for file in val_files:
            os.rename(os.path.join(path, file), os.path.join(val_dir, file))

trainSplit((0.7,0.2,0.1) )
        





