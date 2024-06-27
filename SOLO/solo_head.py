import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial

class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        # initial the two branch head modulelist
        self.cate_head = nn.ModuleList()
        self.ins_head = nn.ModuleList()
        self.cate_out = nn.ModuleList()
        self.ins_out_list = nn.ModuleList()
        
        #-------------------------------------------------------------------
        #lets first make the category branch
        for i in range(self.stacked_convs):  
            self.cate_head.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.seg_feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(num_groups, self.seg_feat_channels),
                    nn.ReLU(inplace=True)
                ))
        #for last layer, its told to use nclasses -1 =>4 and keep bias true
        self.cate_out.append(nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.cate_out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        ))
        self.cate_out = nn.ModuleList(self.cate_out)

        #----------------------------------------------------------------------
        
        #Lets make mask branch
        
        #positional encoding for 1st layer
        self.ins_head.append(
            nn.Sequential(
                nn.Conv2d(self.in_channels + 2, self.seg_feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, self.seg_feat_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        for i in range(1, self.stacked_convs):
            self.ins_head.append(
                nn.Sequential(
                    nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(32, self.seg_feat_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Output layer for mask branch
        self.num_grids_2 = [x * x for x in self.seg_num_grids]
        
        # Add the final output layers to the mask branch
        for out_channels in self.num_grids_2:
            self.ins_out_list.append(
                nn.Sequential(
                    nn.Conv2d(self.seg_feat_channels, out_channels, kernel_size=1, padding=0, bias=True),
                    nn.Sigmoid()
                )
            )
        

    

    # This function initialize weights for head network
    def _init_weights(self):
        ## TODO: initialize the weights
        
        #we will do xavier_uniform initialization
        #if bias exists, weinitialize it to zero
        def xavier_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
        # Apply initialization to all modules
        self.cate_head.apply(xavier_init)
        self.ins_head.apply(xavier_init)
        self.cate_out.apply(xavier_init)
        for layer in self.ins_out_list:
            layer.apply(xavier_init)
        
        


    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        
        #multiapply applies a funcn to set of arguments
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level,new_fpn_list,list(range(len(new_fpn_list))),eval=eval, upsample_shape=quart_shape)
                                                                                                          
        assert len(new_fpn_list) == len(self.seg_num_grids)


        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    
    
    def NewFPN(self, fpn_feat_list):
        
        #forward funcn calls this function. Here we have to interpolate strides for 0th, 4thlayer
        new_fpn_list = [
            F.interpolate(fpn_feat_list[0], scale_factor=0.5),
            fpn_feat_list[1],                                 
            fpn_feat_list[2],                                 
            fpn_feat_list[3],                                 
            F.interpolate(fpn_feat_list[4], size=(25, 34))     
                        ]
        return new_fpn_list
            


    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cate_pred = fpn_feat
        ins_pred  = torch.zeros((fpn_feat.shape[0], fpn_feat.shape[1]+2, fpn_feat.shape[2], fpn_feat.shape[3])).to(device) 
        ins_pred[:,:fpn_feat.shape[1],:,:] = fpn_feat
        #ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid
        #ins_pred = torch.zeros((fpn_feat.shape[0], fpn_feat.shape[1]+2, fpn_feat.shape[2], fpn_feat.shape[3]))
        ####MY CODE#####
        cate_pred = torch.nn.functional.interpolate(cate_pred, (num_grid, num_grid))
        for i, l in enumerate(self.cate_head):
            cate_pred = l(cate_pred) 
        
        cate_pred = self.cate_out(cate_pred)
        #print("num grid:", num_grid)
        #print("FPN shape", fpn_feat.shape)
        #print("Category Prediciton", cate_pred.shape)
        yy,xx = torch.meshgrid([torch.linspace(0.0, 1.0, fpn_feat.shape[2]), torch.linspace(0.0, 1.0, fpn_feat.shape[3])])
        #print("xx:", xx)
        #print("yy:", yy)
        #print(xx.shape)
        #print(yy.shape)
        #print(fpn_feat.shape)
        ins_pred[:,fpn_feat.shape[1],:,:]   = xx
        ins_pred[:,fpn_feat.shape[1]+1,:,:] = yy 
        
        #print(fpn_feat.shape)
        #print(xx.shape)
        #print(yy.shape)
        #print("PRE Ins Pred:", ins_pred.shape)
        #SET ALL LAYERS IN EVAL MODE
        for i, l in enumerate(self.ins_head):
            ins_pred = l(ins_pred)

        ins_pred = torch.nn.functional.interpolate(ins_pred, (fpn_feat.shape[2]*2, fpn_feat.shape[3]*2), 
                                                   mode='bilinear', align_corners=True) #Align Corner?        
        ins_pred = self.ins_out_list[idx](ins_pred)

        print("INS Predictions", ins_pred.shape)
        ################
        # in inference time, upsample the pred to (ori image size/4)
        
        if eval == True:
            ## TODO resize ins_pred 
            ins_pred = torch.nn.functional.interpolate(ins_pred, upsample_shape)
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)

        # check flag
        if eval == False:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        pass



    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        pass

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        pass

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output in training
        featmap_sizes = []
        for i,pred in enumerate(ins_pred_list):
          featmap_sizes.append([ins_pred_list[i].shape[2], ins_pred_list[i].shape[3]] )
        
        featmap_sizes = [featmap_sizes for i in range(len(mask_list))]

        # self.target_single_img(bbox_list[0], label_list[0], mask_list[0], featmap_sizes)
        
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(self.target_single_img,bbox_list,label_list,mask_list, featmap_sizes)
        # check flag
        
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])
        

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image
        h, w = gt_masks_raw.shape[2], gt_masks_raw.shape[3]
        # Area normalized by height and width of image
        area   = torch.sqrt((gt_bboxes_raw[:,2] - gt_bboxes_raw[:,0]) * (gt_bboxes_raw[:,3] - gt_bboxes_raw[:,1]))
        region = torch.zeros((gt_masks_raw.shape[0],4))

        for i in range(gt_masks_raw.shape[0]):
            centre_of_mass = ndimage.measurements.center_of_mass(gt_masks_raw[i,0,:,:].numpy()) 
            # region[i, 0] = centre_of_mass[0]
            # region[i, 1] = centre_of_mass[1]
            region[i, 0] = centre_of_mass[1]
            region[i, 1] = centre_of_mass[0]

        region[:,2] = (gt_bboxes_raw[:,2] - gt_bboxes_raw[:,0]) *0.2/w          #Width 
        region[:,3] = (gt_bboxes_raw[:,3] - gt_bboxes_raw[:,1]) *0.2/h          #Height
        
        #Rescaling Centers
        region[:,0] = region[:,0]/w 
        region[:,1] = region[:,1]/h
        
        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        for i, size in enumerate(featmap_sizes):
            if (i==0):
                idx = area<96
            elif i==1:
                idx = torch.logical_and(area<192,area>48)
            elif i==2: 
                idx = torch.logical_and(area<384,area>96)
            elif i==3: 
                idx = torch.logical_and(area<768,area>192)
            elif i==4: 
                idx = area>=384
            
            if torch.sum(idx) == 0:
                cat_label = torch.zeros((self.seg_num_grids[i],self.seg_num_grids[i]))
                ins_label = torch.zeros((self.seg_num_grids[i]**2, size[0], size[1]))
                ins_index_label = torch.zeros(self.seg_num_grids[i]**2, dtype=torch.bool)

                cate_label_list.append(cat_label)
                ins_label_list.append(ins_label)
                ins_ind_label_list.append(ins_index_label)
                # no_of_zeros += 1
                continue
            
            region_idx = region[idx,:]
            left_ind   = ((region_idx[:,0] - region_idx[:,2]/2)*self.seg_num_grids[i]).int()
            right_ind  = ((region_idx[:,0] + region_idx[:,2]/2)*self.seg_num_grids[i]).int()
            top_ind    = ((region_idx[:,1] - region_idx[:,3]/2)*self.seg_num_grids[i]).int()
            bottom_ind = ((region_idx[:,1] + region_idx[:,3]/2)*self.seg_num_grids[i]).int()

            left   = torch.max(torch.zeros_like(left_ind)                              , left_ind  )
            right  = torch.min(torch.ones_like(right_ind)*(self.seg_num_grids[i] - 1)  , right_ind )
            top    = torch.max(torch.zeros_like(top_ind)                               , top_ind   )
            bottom = torch.min(torch.ones_like(bottom_ind)*(self.seg_num_grids[i] - 1) , bottom_ind)

            xA = torch.max(left    , (region_idx[:,0]*self.seg_num_grids[i]).int() - 1)
            xB = torch.min(right   , (region_idx[:,0]*self.seg_num_grids[i]).int() + 1)
            yA = torch.max(top     , (region_idx[:,1]*self.seg_num_grids[i]).int() - 1)
            yB = torch.min(bottom  , (region_idx[:,1]*self.seg_num_grids[i]).int() + 1)

            #Size of ins_label = S^2 x 2H_feat x 2W_feat
            cat_label = torch.zeros((self.seg_num_grids[i],self.seg_num_grids[i]))

            ins_label = torch.zeros((self.seg_num_grids[i]**2, size[0], size[1]))

            # ins_label = torch.zeros((self.seg_num_grids[i]**2, h, w))
            
            ins_index_label = torch.zeros(self.seg_num_grids[i]**2, dtype=torch.bool)

            mask_interpolate = torch.nn.functional.interpolate(gt_masks_raw[idx,:,:,:],
                                                               (size[0],size[1]))
            mask_interpolate[mask_interpolate > 0.5] = 1
            mask_interpolate[mask_interpolate < 0.5] = 0

            for j in range(xA.size(0)):

              cat_label[yA[j]:yB[j]+1 , xA[j]:xB[j]+1] = gt_labels_raw[idx][j]
              
              flag_matrix = torch.zeros(cat_label.shape)

              flag_matrix[yA[j]:yB[j]+1 , xA[j]:xB[j]+1] = 1
              positive_index = (torch.flatten(flag_matrix) > 0)

              ins_label[positive_index,:,:] = mask_interpolate[j,0,:,:]
              # ins_label[positive_index,:,:] = gt_masks_raw[idx,:,:,:][j,0,:,:]
              ins_index_label = torch.logical_or(ins_index_label,positive_index)

            
            # if ins_index_label.sum() == 0:
            #   print(self.seg_num_grids[i])
            #   print(region_idx[:,1])
            #   print(region_idx[:,3])
            #   # print(((region_idx[:,1] - region_idx[:,3]/2)*self.seg_num_grids[i]).int())
            #   print(xA,xB,yA,yB)
            #   print((flag_matrix >0).sum())
            #   print(flag_matrix[flag_matrix>0])
            #   while(1):
            #     pass

            cate_label_list.append(cat_label)
            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_index_label)
        # check flag
        
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list
    

    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        ## TODO: finish PostProcess
        pass


    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        ## TODO: PostProcess on single image.
        pass

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        pass

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.
        num_pyramids = len(ins_gts_list[0])
        for i in range(num_pyramids):
          plt.figure()
          plt.imshow(img[0,:,:,:].cpu().numpy().transpose(1,2,0))

          if sum(ins_ind_gts_list[0][i]) == 0 :
            # plt.savefig("/pyramid level" + str(i) + ".png")
            plt.show()
            continue
          index = ins_ind_gts_list[0][i] > 0
          label = torch.flatten(cate_gts_list[0][i])[index]
          mask = ins_gts_list[0][i][index,:,:]
          mask = torch.unsqueeze(mask,1)

          reshaped_mask = torch.nn.functional.interpolate(mask,(img.shape[2],img.shape[3]),mode='bilinear')#,align_corners=True)
          
          combined_mask = np.zeros((img.shape[2],img.shape[3],img.shape[1]))

          for idx,l in enumerate(label):
            if l == 1:
              combined_mask[:,:,0] += (reshaped_mask[idx,0,:,:] ).cpu().numpy()
            if l == 2:
              combined_mask[:,:,1] += (reshaped_mask[idx,0,:,:] ).cpu().numpy()
            if l == 3:
              combined_mask[:,:,2] += (reshaped_mask[idx,0,:,:] ).cpu().numpy()
          
          origin_img = img[0,:,:,:].cpu().numpy().transpose(1,2,0)
          index_to_mask = np.where(combined_mask > 0)
          masked_image = copy.deepcopy(origin_img)
          masked_image[index_to_mask[0],index_to_mask[1],:] = 0

          mask_to_plot = (combined_mask + masked_image)
          plt.imshow(mask_to_plot)#,alpha = 1)
          plt.show()

        

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        pass

from backbone import *
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
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

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        img = img.float()
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,bbox_list,label_list,mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)


