## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list

    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        
        #we use this function to make our logic of getting right masks together
        
        #path is a list that contains path to images,masks etc
        #images and masks are h5 file. labels and bbox are npy files
        
        self.images = h5py.File(path[0],'r')['data'] #(3265,3,300,400)
        self.mask   = h5py.File(path[1],'r')['data'] #(3843,300,400) ->3d
        self.label = np.load(path[2], allow_pickle=True) #3265, ->1d
        self.bbox   = np.load(path[3], allow_pickle=True) #3265, ->1d
        
        #we go over each label. see its length. we take that length elements from mask and append into final_mask as one item
        self.corrected_mask = []
        i = 0 
        mask_shape = self.mask[0].shape
        for l in range(self.label.shape[0]):
            length = self.label[l].size
            mask_img = []
            for j in range(length):
                mask_img.append(self.mask[i,:,:])
                i+=1
                
            self.corrected_mask.append(mask_img)
        
        
    def __getitem__(self, index):
        # TODO: __getitem__
        
        #images is 4dimensional
        img = self.images[index, ...]
        #ourimage = ourimages[1,...]
        #corrected_mask is 1d of shape 3265
        mask = self.corrected_mask[index]
        #ourmask = corrected_mask[1]
        #bbox is 1 dimensional
        bbox = self.bbox[index]
        #ourbb = ourbbox[1]
        #label is also 1d array -> convert to tensors
        label_index  = self.label[index]
        #ourlabel = ourlabels[1]
        label_index = torch.tensor(label_index,dtype = torch.float)
        #torch.tensor(ourlabel,dtype = torch.float)
        
        
        #img,mask,bbox needs to be transformed before checking assert below
        transed_img,transed_mask,transed_bbox = self.pre_process_batch(img,mask,bbox)    

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label_index, transed_mask, transed_bbox
    
    
    def __len__(self):
        return self.images.shape[0]

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
        
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        
        #currently image is of type _h1.dataset.Dataset -> convert to tensor
        #normalize each channel
        #reside image to 800,1088 from 300,400
        img = torch.from_numpy(np.array(img).astype(np.float))
        img = img / 255.0
        img =  torch.unsqueeze(img, 0)
        img  = torch.nn.functional.interpolate(img, size=(800, 1088), mode='bilinear') #bilinear interpolation as taught in class
        img = transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))(img[0])  #since we added a dimension by unsqueezing, we apply normalization on last 3 dimensions

        
        
        #bounding box pre processing ->since images are resized, box coordinates also needs to be resized
        #bbox is 2dimensional of shape n_box,4
        bbox = torch.tensor(bbox)
        #since x coordinates of image went from 300 ->800, bbox needs to be scaled accordingly
        #since y coordinates of image went from 400 -> 1066, bbox needs to be scaled accordingly
        bbox[:,0] = bbox[:,0] * (800/300)
        bbox[:,1] = bbox[:,1] * (1066/400)
        bbox[:,2] = bbox[:,2] * (800/300)
        bbox[:,3] = bbox[:,3] * (1066/400)
        bbox[:,0] = bbox[:,0] + 11 
        bbox[:,2] = bbox[:,2] + 11 
        
                
        #mask pre processing
        #same operations as image
        mask = torch.tensor(np.array(mask).astype(np.float), dtype=torch.float)
        mask = torch.unsqueeze(mask,0)
        mask = torch.nn.functional.interpolate(mask, size=(800, 1088), mode='bilinear') 
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] =  0
        

        # check flag
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.squeeze(0).shape[0]
        
        mask = torch.squeeze(mask,0) #to satisy asserts in getitem and preprocess both
        return img, mask, bbox

        

class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        transed_img_list, label_list, transed_mask_list, transed_bbox_list = list(zip(*batch))
        # Ensure that each label is 1-dimensional and convert the tuple to a list
        #earlier we were getting 3d labels but generate targets function expected 1d labels
        #also we were initially returning bboxes and mask as tuples instead of lists which was expected
        #so we modified our collect_fn
        label_list = [torch.flatten(label) for label in label_list]
        
        # Convert masks and boxes tuples to lists
        transed_mask_list = list(transed_mask_list)
        transed_bbox_list = list(transed_bbox_list)
        
        return torch.stack(transed_img_list, dim=0), label_list, transed_mask_list, transed_bbox_list


        
        

    def loader(self):
        # TODO: return a dataloader
        return torch.utils.data.DataLoader(self.dataset,shuffle = self.shuffle,num_workers = self.num_workers,batch_size = self.batch_size,collate_fn = self.collect_fn)

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    
    color_dict = {
    1: (0, 0, 350/255),   
    2: (0, 1, 0),  
    3: (1, 0, 0)      }
    

    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label, mask, bbox = [data[i] for i in range(len(data))]
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size
    
        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]
        

        for i in range(batch_size):
            bbox_list = bbox[i]
            mask_list = mask[i]
            img_height, img_width = img.shape[2], img.shape[3]
            
            #Print image
            fig, ax = plt.subplots(figsize=(10, 10))
            #since imshow takes h,w,channels, we use permute
            ax.imshow(img[i].permute(1, 2, 0))
            
            #Printt bbox
            for bb in bbox_list:
                bb = bb.cpu().numpy()
                x1, y1, x2, y2 = bb
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            
            #Print masks
            masks_indiv = []
            for j, mask_tensor in enumerate(mask_list):
                #mask_tensor = 1-mask_tensor
                masks_indiv.append(mask_tensor)
                mask_array = mask_tensor.cpu().numpy()
                #color the map
                class_id = int(label[i][j])  
                mask_color = color_dict.get(class_id, (255, 255, 255)) 
                
                rgba_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 4))
                for c in range(3):
                    rgba_mask[:, :, c] = mask_array * mask_color[c]
                rgba_mask[:, :, 3] = mask_array
                
                ax.imshow(rgba_mask,alpha =0.8)
                plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
                plt.show()

        if iter == 3:
            break
    
#car (label1) - blue
#human(label2) - green
#animal(label3)- red
            
            

