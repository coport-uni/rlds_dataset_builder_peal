import numpy as np
import tqdm
import os
import natsort
import string
import cv2

class DatasetLoader():

    def __init__(self):
        # Load images from folder and sort by name
        print("dataset converter")

    def parser(self, input):
        ''' 
        String type array to array 
        '''
        a = input.split()
        num = int(a[0].split("\D+")[0].replace("[","").replace(",","").replace("]",""))
        array = np.array([float(a[1].split("\D+")[0].replace("[","").replace(",","").replace("]","").replace(".jpg","")), \
        float(a[2].split("\D+")[0].replace("[","").replace(",","").replace("]","").replace(".jpg","")), \
        float(a[3].split("\D+")[0].replace("[","").replace(",","").replace("]","").replace(".jpg","")), \
        float(a[4].split("\D+")[0].replace("[","").replace(",","").replace("]","").replace(".jpg","")), \
        float(a[5].split("\D+")[0].replace("[","").replace(",","").replace("]","").replace(".jpg","")), \
        float(a[6].split("\D+")[0].replace("[","").replace(",","").replace("]","").replace(".jpg",""))])

        return(array)

    def mapper(self, array, mode):
        ''' 
        Mapping various type array to -1 ~ +1 array 
        '''
        # mode 1 = joint
        # mode 2 = eef
        new_array = [0,0,0,0,0,0]

        if mode == "joint":
            new_array[0] = round(np.interp(array[0], [-90,90], [-1,1]),0)
            new_array[1] = round(np.interp(array[1], [-75,0], [-1,1]),0)
            new_array[2] = round(np.interp(array[2], [0,-145], [-1,1]),0)
            new_array[3] = round(np.interp(array[3], [-90,100], [-1,1]),0)
            new_array[4] = round(np.interp(array[4], [-75,90], [-1,1]),0)
            new_array[5] = round(np.interp(array[5], [90,-90], [-1,1]),0)

        elif mode == "eef":
            # x,y,z,u,v,w
            new_array[0] = round(np.interp(array[0], [700,300], [-1,1]),0)
            new_array[1] = round(np.interp(array[1], [-620,620], [-1,1]),0)
            new_array[2] = round(np.interp(array[2], [880,190], [-1,1]),0)
            new_array[3] = round(np.interp(array[3], [-180, 180], [-1,1]),0)
            new_array[4] = round(np.interp(array[4], [-180, 180], [-1,1]),0)
            new_array[5] = round(np.interp(array[5], [-180, 180], [-1,1]),0)

        else:
            print("ErrorOnMapper!")

        return new_array

    def load_data(self, path):

        action_array = []
        eximage_array = []
        state_array = []
        inimage_array = []

        self.path_ex = path + "/external"
        self.path_in = path + "/internal"
        self.list_ex = natsort.natsorted(os.listdir(self.path_ex))
        self.list_in = natsort.natsorted(os.listdir(self.path_in))
        self.list_ex_len = len(self.list_ex)
        self.list_in_len = len(self.list_in)

        for item in tqdm.tqdm(self.list_ex,desc= "Loading state & eximage"):
            item_state = self.parser(item)
            item_state = self.mapper(item_state, "joint")
            item_eximage = cv2.imread(self.path_ex + "/" + item)
            item_eximage = cv2.cvtColor(item_eximage, cv2.COLOR_BGR2RGB)
            item_eximage = cv2.resize(item_eximage,dsize=(256,256))

            state_array.append(item_state)
            eximage_array.append(item_eximage)

        for item in tqdm.tqdm(self.list_in,desc= "Loading action & inimage"):
            item_action = self.parser(item)
            item_action = self.mapper(item_action, "eef")
            # item_inimage = cv2.imread(self.path_ex + "/" + item)
            # item_inimage = cv2.cvtColor(item_inimage, cv2.COLOR_BGR2RGB)
            # item_inimage = cv2.resize(item_inimage,dsize=(256,256))

            item_action = np.append(item_action, [0])

            action_array.append(item_action)
            # inimage_array.append(item_inimage)

        return action_array, eximage_array, state_array, self.list_ex_len

    def create_episode(self, path, image, state, action, reward, first, prompt):
        episode = []
        episode.append({
            'image': np.array(image, dtype=np.uint8),
            'state': np.array(state, dtype=np.float32),
            'action': np.array(action, dtype=np.float32),
            'reward': bool(reward),
            'first': bool(first),
            'language_instruction': prompt,
        })
        np.save(path, episode)

def main():
    ds= DatasetLoader()
    prompt = "move arm to front side of sliver-colored grinder"
    data_num = 9

    os.makedirs('/workspace/RLDS/rlds_dataset_builder/NeuromekaNet/data/train', exist_ok=True)
    os.makedirs('/workspace/RLDS/rlds_dataset_builder/NeuromekaNet/data/val', exist_ok=True)

    # Define number of dataset
    for j in tqdm.tqdm(range(data_num),desc= "Making TFDS"):
        path = f"/workspace/RLDS/rlds_dataset_builder/NeuromekaNet/test_data_v3/test_dataset_{j}"
        action, eximage, state, ex_num = ds.load_data(path)
        # Define number of steps in episode
        for i in range(ex_num):
            # Validation
            if j == 0 or j == 2:
                # terminal 
                if i == ex_num - 1:
                    ds.create_episode(f'data/val/episode_{j}_{i}.npy', eximage[i], state[i], action[i], 1, 0, prompt)
                    print("val_last")
                # first
                elif i == 0:
                    ds.create_episode(f'data/val/episode_{j}_{i}.npy', eximage[i], state[i], action[i], 0, 1, prompt)
                    print("val_first")
                # middle
                else:
                    ds.create_episode(f'data/val/episode_{j}_{i}.npy', eximage[i], state[i], action[i], 0, 0, prompt)

            else:
                if i == ex_num - 1:
                    ds.create_episode(f'data/train/episode_{j}_{i}.npy', eximage[i], state[i], action[i], 1, 0, prompt)

                elif i == 0:
                    ds.create_episode(f'data/train/episode_{j}_{i}.npy', eximage[i], state[i], action[i], 0, 1, prompt)

                else:
                    ds.create_episode(f'data/train/episode_{j}_{i}.npy', eximage[i], state[i], action[i], 0, 0, prompt)

    print("complete")


if __name__ == '__main__':
    main()