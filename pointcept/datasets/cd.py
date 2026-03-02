"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors, KDTree
import pickle
from torch_cluster import grid_cluster
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
import random
from  pointcept.utils.voxelize import voxelize
from torch_geometric.nn import knn
import re

@DATASETS.register_module()
class CDDataset(Dataset):
  
    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        pre_split="PRE",
        radius=50,
        voxel_size=0.5,
        nameInPly= "vertex",
        sample_num=1000,
        num_classes=7,
        loop=1,
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
    ):

        super(CDDataset, self).__init__()
        self.transform = Compose(transform)
        
        self.file_root = os.path.join(data_root,split)
        self.pre_dir = os.path.join(data_root,pre_split)
        self.pre_dir = os.path.join(self.pre_dir,split)

        self.voxel_size = voxel_size
        self.radius = radius
        self.voxel_size=voxel_size
        self.nameInPly=nameInPly
        self.sample_num=sample_num
        self._get_paths()
        self.sum_num_class = torch.zeros(num_classes)
        self.process()  #预处理:读取所有点，建立tree
        # if self.sum_num_class.sum() == 0:
        #    self.get_sum_num_class()
        # self.weight_classes = 1 - self.sum_num_class / self.sum_num_class.sum() #?
        self.centers()
      
        
        
    def centers(self):
        self._centres_for_sampling = []
        r=self.radius/10
        for idx in range(len(self.filesPC0)):
            pc1 = torch.load(os.path.join(self.pre_dir, 'pc1_{}.pt'.format(idx)))
            coords = torch.round((pc1.pos) / r)
            cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
            cluster, unique_pos_indices = consecutive_cluster(cluster)
            #mode=last
            # pc1["pos"] = pc1.pos[unique_pos_indices]
            # pc1["cd"]= pc1.cd[unique_pos_indices]
 
            item_min = pc1.cd.min()
            pc1.cd = F.one_hot(pc1.cd - item_min)
            pc1.cd = scatter_add(pc1.cd, cluster, dim=0)
            pc1["cd"] = pc1.cd.argmax(dim=-1) + item_min
            pc1["pos"] = scatter_mean(pc1.pos, cluster, dim=0)
            
            centres = torch.empty((pc1.pos.shape[0], 5), dtype=torch.float)
            centres[:, :3] = pc1.pos
            centres[:, 3] = idx
            centres[:, 4] = pc1.cd
            self._centres_for_sampling.append(centres)
        
        self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
        uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
        uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
        self._label_counts = uni_counts / np.sum(uni_counts)
        self._labels = uni
        self.weight_classes = torch.from_numpy(self._label_counts).type(torch.float)
        print(self.weight_classes)
       
    
    def load_tree(self,i):
        tree_dir=os.path.join(self.pre_dir,"KDTREE")
        namet0=os.path.basename(self.filesPC0[i]).split(".")[0]+ "_"+ str(int(i)) + ".p"
        path0=os.path.join(tree_dir,namet0)
        file = open(path0, "rb")
        tree0 = pickle.load(file)
        file.close()
        namet1=os.path.basename(self.filesPC1[i]).split(".")[0]+ "_"+ str(int(i)) + ".p"
        path1=os.path.join(tree_dir,namet1)
        file = open(path1, "rb")
        tree1 = pickle.load(file)
        file.close()
        return tree0,tree1    





    def get_sum_num_class(self):
        for idx in range(len(self.filesPC0)):
            pc1 = torch.load(os.path.join(self.pre_dir, 'pc1_{}.pt'.format(idx)))
            cpt = torch.bincount(pc1.cd)
            for c in range(cpt.shape[0]):
                self.sum_num_class[c] += cpt[c]


    def process(self):
        for idx in range(len(self.filesPC0)):
            exist_file=os.path.isfile(os.path.join(self.pre_dir, 'pc0_{}.pt'.format(idx)))
            exist_file=os.path.isfile(os.path.join(self.pre_dir, 'pc1_{}.pt'.format(idx)))
        if not exist_file:
            #tree
            tree_dir=os.path.join(self.pre_dir,"KDTREE")
            if not os.path.exists(os.path.join(tree_dir)):
                os.makedirs(tree_dir)
            
            for idx in range(len(self.filesPC0)):
                print("process",self.filesPC0[idx])
                vertex0=self.read_from_ply_PC0(self.filesPC0[idx])
                vertex1=self.read_from_ply_PC1(self.filesPC1[idx])
                coord0=torch.from_numpy(vertex0[:,:3])
                coord1 = torch.from_numpy(vertex1[:,:3])
                label_cd=torch.from_numpy(vertex1[:,3]).to(dtype=torch.long)
                label_seg_t1=torch.from_numpy(vertex1[:,4]).to(dtype=torch.long)
                label_seg_t0=torch.from_numpy(vertex0[:,3]).to(dtype=torch.long)
                pc0 = Data(pos=coord0,seg=label_seg_t0)
                pc1 = Data(pos=coord1,seg=label_seg_t1, cd=label_cd)
                cpt = torch.bincount(pc1.cd)
                for c in range(cpt.shape[0]):
                    self.sum_num_class[c] += cpt[c]
                torch.save(pc0, os.path.join(self.pre_dir, 'pc0_{}.pt'.format(idx)))
                torch.save(pc1, os.path.join(self.pre_dir, 'pc1_{}.pt'.format(idx)))
                
                #tree
                namet0=os.path.basename(self.filesPC0[idx]).split(".")[0]+ "_"+ str(idx) + ".p"
                path0=os.path.join(tree_dir,namet0)
                if not os.path.isfile(path0):
                    tree=KDTree(np.asarray(coord0[:,:-1]),leaf_size=10)
                    file=open(path0,"wb")
                    pickle.dump(tree,file)
                    file.close()
                
                namet1=os.path.basename(self.filesPC1[idx]).split(".")[0]+ "_"+str(idx) + ".p"
                path1=os.path.join(tree_dir,namet1)
                if not os.path.isfile(path1):
                    tree=KDTree(np.asarray(coord1[:,:-1]),leaf_size=10)
                    file=open(path1,"wb")
                    pickle.dump(tree,file)
                    file.close()
               


    def _get_paths(self):
        self.filesPC0 = []
        self.filesPC1 = []
        globPath = os.scandir(self.file_root)
        for dir in globPath:
            if dir.is_dir():
                curDir = os.scandir(dir)
                for f in curDir:
                    if f.name == "pointCloud0.ply":
                        self.filesPC0.append(f.path)
                    elif f.name == "pointCloud1.ply":
                        self.filesPC1.append(f.path)
                curDir.close()
        globPath.close()


     
    def __getitem__(self, idx):
   
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        file_idx = centre[3].int()
        centre=centre[:3]
        c_centre=centre[:-1]
        c_centre=np.asarray(c_centre)
        c_centre = np.expand_dims(c_centre, 0)


        #数据加载
        pc0 = torch.load(os.path.join(self.pre_dir, 'pc0_{}.pt'.format(file_idx)))
        pc1 = torch.load(os.path.join(self.pre_dir, 'pc1_{}.pt'.format(file_idx)))
        tree0,tree1=self.load_tree(file_idx)
        
        pc0_s=Data()

        pc1_s=Data()
        indice0 = torch.LongTensor(tree0.query_radius(c_centre, r=self.radius)[0])
        
        keys0=["pos","seg"]
        for key in enumerate(keys0):
            item = pc0[key[1]]
            item = item[indice0]
            setattr(pc0_s, key[1], item)
        keys1=["pos","seg","cd"]
        indice1 = torch.LongTensor(tree1.query_radius(c_centre, r=self.radius)[0])
        for key in enumerate(keys1):
            item = pc1[key[1]]
            item = item[indice1]
            setattr(pc1_s, key[1], item)
        
        coord0=pc0_s.pos
        coord1 = pc1_s.pos
       
   
        if coord0.shape[0]<=1024:
            return self.__getitem__(30)
        if coord1.shape[0]<=1024:
            return self.__getitem__(30)

        label_ch=pc1_s.cd
        label_seg_t1=pc1_s.seg
        label_seg_t0=pc0_s.seg
        
     
        #normalize
        min0 = torch.unsqueeze(coord0.min(0)[0], 0)

        coord0[:, 0] = (coord0[:, 0] - min0[0,0])  # x
        coord0[:, 1] = (coord0[:, 1] - min0[0,1])  # y
        coord0[:, 2] = (coord0[:, 2] - min0[0,2])  # z
        coord1[:, 0] = (coord1[:, 0] - min0[0,0])  # x
        coord1[:, 1] = (coord1[:, 1] - min0[0,1])  # y
        coord1[:, 2] = (coord1[:, 2] - min0[0,2])  # z
        feat0=coord0.clone()
        feat1=coord1.clone()
    
        if self.voxel_size>0:
            uniq_idx0 = voxelize(coord0.numpy(), self.voxel_size)
            coord0, feat0,label_seg_t0= coord0[uniq_idx0], feat0[uniq_idx0],label_seg_t0[uniq_idx0]
            uniq_idx1 = voxelize(coord1.numpy(), self.voxel_size)
            coord1, feat1,label_cd,label_seg_t1= coord1[uniq_idx1], feat1[uniq_idx1],label_ch[uniq_idx1],label_seg_t1[uniq_idx1]
 
        #temporal 

        offset_t0=coord0.shape[0]
        offset_t1=coord1.shape[0]
        coord=torch.cat([coord0,coord1],dim=0)
        seg_cd0=torch.full([coord0.shape[0]],-1)
        seg_cd=torch.cat([seg_cd0,label_cd],dim=0)
        segment=torch.cat([label_seg_t0,label_seg_t1],dim=0)

        ft0=torch.zeros(coord0.shape[0])   
        ft1=torch.ones(coord1.shape[0])   
        ft=torch.cat([ft0,ft1],dim=0).reshape(-1,1)
       
        feat=torch.cat([feat0,feat1],dim=0)
        
        featt=torch.cat([feat,ft],dim=1)
   

        data_dict = dict(
            coord=coord,
            feat=featt, #featt
            segment_cd=seg_cd,#seg_cd
            segment=segment,
            offset=coord.shape[0],
        )

        return data_dict
 
    
    def read_from_ply_PC0(self,filename):
        """read XYZ for each vertex."""
        nameInPly=self.nameInPly
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
            vertices[:, 0] = plydata[nameInPly].data["x"]
            vertices[:, 1] = plydata[nameInPly].data["y"]
            vertices[:, 2] = plydata[nameInPly].data["z"]
            vertices[:, 3] = plydata[nameInPly].data["label_mono"]
        return vertices
    
    def read_from_ply_PC1(self,filename):
        """read XYZ for each vertex."""
        nameInPly=self.nameInPly
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices = np.zeros(shape=[num_verts, 5], dtype=np.float32)
            vertices[:, 0] = plydata[nameInPly].data["x"]
            vertices[:, 1] = plydata[nameInPly].data["y"]
            vertices[:, 2] = plydata[nameInPly].data["z"]
            vertices[:, 3] = plydata[nameInPly].data["label_ch"]
            vertices[:, 4] = plydata[nameInPly].data["label_mono"]
        return vertices
     

    # def __getitem__(self, idx):
    #     if self.test_mode:
    #         return self.prepare_test_data(idx)
    #     else:
    #         return self.prepare_train_data(idx)

    def __len__(self):
        return self.sample_num


@DATASETS.register_module()
class CDDatasetNYC(Dataset):
  
    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        pre_split="PRE",
        radius=50,
        voxel_size=0.5,
        nameInPly= "vertex",
        sample_num=1000,
        num_classes=7,
        loop=1,
        test_mode=False,
        test_cfg=None,
        cache=False,
    ):

        super(CDDatasetNYC, self).__init__()

        
        self.file_root = os.path.join(data_root,split)
        self.pre_dir = os.path.join(data_root,pre_split)
        self.pre_dir = os.path.join(self.pre_dir,split)

        self.voxel_size = voxel_size
        self.radius = radius
        self.voxel_size=voxel_size
        self.nameInPly=nameInPly
        self.sample_num=sample_num
        self._get_paths()
        if(self.sample_num>0):
            self.sum_num_class = torch.zeros(num_classes)
            self.process() 
            self.centers()
      
        
        
    def centers(self):
        self._centres_for_sampling = []
        centres_for_sampling0=[]
      

        r=self.radius/10
        
        for idx in range(len(self.filesPC0)):
            pc1 = torch.load(os.path.join(self.pre_dir, 'pc1_{}.pt'.format(idx)))
            coords = torch.round((pc1.pos) / r)
            cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
            cluster, unique_pos_indices = consecutive_cluster(cluster)
           
            item_min = pc1.cd.min()
            pc1.cd = F.one_hot(pc1.cd - item_min)
            pc1.cd = scatter_add(pc1.cd, cluster, dim=0)
            pc1["cd"] = pc1.cd.argmax(dim=-1) + item_min
            pc1["pos"] = scatter_mean(pc1.pos, cluster, dim=0)
            
            centres = torch.empty((pc1.pos.shape[0], 5), dtype=torch.float)
            centres[:, :3] = pc1.pos
            centres[:, 3] = idx
            centres[:, 4] = pc1.cd
            self._centres_for_sampling.append(centres)
        self._centres_for_sampling = torch.cat(self._centres_for_sampling, dim=0)
        self._centres_for_sampling=self._centres_for_sampling[self._centres_for_sampling[:,4]<4]
        uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
        uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
        self._label_counts = uni_counts / np.sum(uni_counts)
        self._labels = uni
        self.weight_classes = torch.from_numpy(self._label_counts).type(torch.float)
        print(self.weight_classes)
      
    
    def load_tree(self,i):
        tree_dir=os.path.join(self.pre_dir,"KDTREE")
        namet0=os.path.basename(self.filesPC0[i]).split(".")[0]+ "_"+ str(int(i)) + ".p"
        path0=os.path.join(tree_dir,namet0)
        file = open(path0, "rb")
        tree0 = pickle.load(file)
        file.close()
        namet1=os.path.basename(self.filesPC1[i]).split(".")[0]+ "_"+ str(int(i)) + ".p"
        path1=os.path.join(tree_dir,namet1)
        file = open(path1, "rb")
        tree1 = pickle.load(file)
        file.close()
        return tree0,tree1    





    def get_sum_num_class(self):
        for idx in range(len(self.filesPC0)):
            pc1 = torch.load(os.path.join(self.pre_dir, 'pc1_{}.pt'.format(idx)))
            cpt = torch.bincount(pc1.cd)
            for c in range(cpt.shape[0]):
                self.sum_num_class[c] += cpt[c]


    def process(self):
        for idx in range(len(self.filesPC0)):
            exist_file=os.path.isfile(os.path.join(self.pre_dir, 'pc0_{}.pt'.format(idx)))
            exist_file=os.path.isfile(os.path.join(self.pre_dir, 'pc1_{}.pt'.format(idx)))
        if not exist_file:
            #tree
            tree_dir=os.path.join(self.pre_dir,"KDTREE")
            if not os.path.exists(os.path.join(tree_dir)):
                os.makedirs(tree_dir)
            
            for idx in range(len(self.filesPC0)):
                print("processing",self.filesPC0[idx])
                vertex0=self.read_from_ply_PC0(self.filesPC0[idx])
                vertex1=self.read_from_ply_PC(self.filesPC1[idx])
                coord0=torch.from_numpy(vertex0[:,:3])
                coord1 = torch.from_numpy(vertex1[:,:3])
                label_seg_0=torch.from_numpy(vertex0[:,3])#rgb -> labelmoni
                label_seg_1=torch.from_numpy(vertex1[:,3])
                label_cd=torch.from_numpy(vertex1[:,4]).to(dtype=torch.long)

                pc0 = Data(pos=coord0,seg=label_seg_0)
                pc1 = Data(pos=coord1,seg=label_seg_1,cd=label_cd)
    
                torch.save(pc0, os.path.join(self.pre_dir, 'pc0_{}.pt'.format(idx)))
                torch.save(pc1, os.path.join(self.pre_dir, 'pc1_{}.pt'.format(idx)))
                
                #tree
                namet0=os.path.basename(self.filesPC0[idx]).split(".")[0]+ "_"+ str(idx) + ".p"
                path0=os.path.join(tree_dir,namet0)
                if not os.path.isfile(path0):
                    tree=KDTree(np.asarray(coord0[:,:-1]),leaf_size=10)
                    file=open(path0,"wb")
                    pickle.dump(tree,file)
                    file.close()
                
                namet1=os.path.basename(self.filesPC1[idx]).split(".")[0]+ "_"+str(idx) + ".p"
                path1=os.path.join(tree_dir,namet1)
                if not os.path.isfile(path1):
                    tree=KDTree(np.asarray(coord1[:,:-1]),leaf_size=10)
                    file=open(path1,"wb")
                    pickle.dump(tree,file)
                    file.close()
               


    def _get_paths(self):
        self.filesPC0 = []
        self.filesPC1 = []
        globPath = os.scandir(self.file_root)
        for dir in globPath:
            if dir.is_dir():
                curDir = os.scandir(dir) 
                pc0_path = None
                pc1_path = None
                for f in curDir:
                    if f.name == "pointCloud0.ply":
                        pc0_path = f.path
                    elif f.name == "pointCloud1.ply":
                        pc1_path = f.path
                if pc0_path is not None and pc1_path is not None:
                    self.filesPC0.append(pc0_path)  
                    self.filesPC1.append(pc1_path)  
                curDir.close()
        globPath.close()


     
    def __getitem__(self, idx):
        if(self.sample_num>0):
            data=self.getitemrandom()
        else:
            data=self.getitemseq(idx)
        return data
 
    
    def read_from_ply_PC(self,filename):
        """read XYZ for each vertex."""
        nameInPly=self.nameInPly
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices = np.zeros(shape=[num_verts, 5], dtype=np.float32)
            vertices[:, 0] = plydata[nameInPly].data["x"]
            vertices[:, 1] = plydata[nameInPly].data["y"]
            vertices[:, 2] = plydata[nameInPly].data["z"]
            vertices[:, 3] = plydata[nameInPly].data["label_mono"]
            vertices[:, 4] = plydata[nameInPly].data["label_ch"]
        return vertices
    def read_from_ply_PC0(self,filename):
        """read XYZ for each vertex."""
        nameInPly=self.nameInPly
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices = np.zeros(shape=[num_verts, 4], dtype=np.float32)
            vertices[:, 0] = plydata[nameInPly].data["x"]
            vertices[:, 1] = plydata[nameInPly].data["y"]
            vertices[:, 2] = plydata[nameInPly].data["z"]
            vertices[:, 3] = plydata[nameInPly].data["label_mono"]
        return vertices

    def getitemrandom(self):
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        file_idx = centre[3].int()
        centre=centre[:3]
        c_centre=centre[:-1]
        c_centre=np.asarray(c_centre)
        c_centre = np.expand_dims(c_centre, 0)



        pc0 = torch.load(os.path.join(self.pre_dir, 'pc0_{}.pt'.format(file_idx)))
        pc1 = torch.load(os.path.join(self.pre_dir, 'pc1_{}.pt'.format(file_idx)))
        tree0,tree1=self.load_tree(file_idx)
        
        pc0_s=Data()

        pc1_s=Data()
        indice0 = torch.LongTensor(tree0.query_radius(c_centre, r=self.radius)[0])
        
        keys0=["pos","seg"]
        for key in enumerate(keys0):
            item = pc0[key[1]]
            item = item[indice0]
            setattr(pc0_s, key[1], item)
        keys1=["pos","seg","cd"]
        indice1 = torch.LongTensor(tree1.query_radius(c_centre, r=self.radius)[0])
        for key in enumerate(keys1):
            item = pc1[key[1]]
            item = item[indice1]
            setattr(pc1_s, key[1], item)
        
        coord0=pc0_s.pos
        coord1 = pc1_s.pos

        
        if (coord0.shape[0]+coord1.shape[0])<=1024:
            print("d")
            return self.__getitem__(66)

        c=torch.cat([coord0,coord1],dim=0)
        label_cd_t1=pc1_s.cd
        label_seg_0=pc0_s.seg
        label_seg_1=pc1_s.seg

     
        #normalize
        min0 = torch.unsqueeze(c.min(0)[0], 0)

        coord0[:, 0] = (coord0[:, 0] - min0[0,0])  # x
        coord0[:, 1] = (coord0[:, 1] - min0[0,1])  # y
        coord0[:, 2] = (coord0[:, 2] - min0[0,2])  # z
        coord1[:, 0] = (coord1[:, 0] - min0[0,0])  # x
        coord1[:, 1] = (coord1[:, 1] - min0[0,1])  # y
        coord1[:, 2] = (coord1[:, 2] - min0[0,2])  # z
   
    
        if self.voxel_size>0:
            uniq_idx0 = voxelize(coord0.numpy(), self.voxel_size)
            coord0, label_seg_0= coord0[uniq_idx0], label_seg_0[uniq_idx0]
            uniq_idx1 = voxelize(coord1.numpy(), self.voxel_size)
            coord1, label_seg_1,label_cd_t1= coord1[uniq_idx1], label_seg_1[uniq_idx1],label_cd_t1[uniq_idx1]
 
        #temporal 

        offset_t0=coord0.shape[0]
        offset_t1=coord1.shape[0]
        coord=torch.cat([coord0,coord1],dim=0)
        seg_cd0=torch.full([coord0.shape[0]],-1)
        segment_cd=torch.cat([seg_cd0,label_cd_t1],dim=0)

        ft0=torch.zeros(coord0.shape[0])   
        ft1=torch.ones(coord1.shape[0])   
        ft=torch.cat([ft0,ft1],dim=0).reshape(-1,1)
        label_seg=torch.cat([label_seg_0,label_seg_1],dim=0)#semantic
        mask=label_seg>3
        label_seg[mask]=-1


        featt=torch.cat([coord,ft],dim=1)
        
 
        mask=segment_cd>3
        segment_cd[mask]=-1
        segment_cd[segment_cd<0]=-1


        if coord.shape[0]>42000:
            print("1")
            return self.__getitem__(random.randint(0,self.sample_num))


        data_dict = dict(
            coord=coord,
            feat=featt, #featt
            segment=label_seg.long(),
            segment_cd=segment_cd,
            offset=coord.shape[0],
        )

        return data_dict
    def getitemseq(self,idx):
 
        vertex0=self.read_from_ply_PC0(self.filesPC0[idx])
        vertex1=self.read_from_ply_PC(self.filesPC1[idx])

        coord0=torch.from_numpy(vertex0[:,:3])
        coord1 = torch.from_numpy(vertex1[:,:3])
        label_seg_0=torch.from_numpy(vertex0[:,3])
        label_seg_1=torch.from_numpy(vertex1[:,3])
        label_cd_t0= torch.full([coord0.shape[0]],-1)
        label_cd_t1=torch.from_numpy(vertex1[:,4]).to(dtype=torch.long)

     
        #normalize
        min0 = torch.unsqueeze(coord0.min(0)[0], 0)

        coord0[:, 0] = (coord0[:, 0] - min0[0,0])  # x
        coord0[:, 1] = (coord0[:, 1] - min0[0,1])  # y
        coord0[:, 2] = (coord0[:, 2] - min0[0,2])  # z
        coord1[:, 0] = (coord1[:, 0] - min0[0,0])  # x
        coord1[:, 1] = (coord1[:, 1] - min0[0,1])  # y
        coord1[:, 2] = (coord1[:, 2] - min0[0,2])  # z
   
    
        if self.voxel_size>0:
            uniq_idx0 = voxelize(coord0.numpy(), self.voxel_size)
            coord0, label_seg_0,label_cd_t0= coord0[uniq_idx0], label_seg_0[uniq_idx0],label_cd_t0[uniq_idx0]
            uniq_idx1 = voxelize(coord1.numpy(), self.voxel_size)
            coord1, label_seg_1,label_cd_t1= coord1[uniq_idx1], label_seg_1[uniq_idx1],label_cd_t1[uniq_idx1]
 

        #temporal 

        offset_t0=coord0.shape[0]
        offset_t1=coord1.shape[0]
        coord=torch.cat([coord0,coord1],dim=0)
        segment_cd=torch.cat([label_cd_t0,label_cd_t1],dim=0)

        ft0=torch.zeros(coord0.shape[0])   
        ft1=torch.ones(coord1.shape[0])   
        ft=torch.cat([ft0,ft1],dim=0).reshape(-1,1)
        label_seg=torch.cat([label_seg_0,label_seg_1],dim=0)
        mask=label_seg>3
        label_seg[mask]=-1
  
        featt=torch.cat([coord,ft],dim=1)
        

        mask=segment_cd>3
        segment_cd[mask]=-1
       

        data_dict = dict(
            coord=coord,
            feat=featt, #featt
            segment=label_seg.long(),
            segment_cd=segment_cd,
            offset=coord.shape[0],
        )


        return data_dict
  

    def __len__(self):
        if(self.sample_num>0):
            num= self.sample_num
        else:
            num=len(self.filesPC0)
        return num

@DATASETS.register_module()
class CDDatasetStreet(Dataset):
  
    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        pre_split="PRE",
        radius=50,
        voxel_size=0.5,
        nameInPly= "vertex",
        sample_num=1000,
        num_classes=7,
        loop=1,
        test_mode=False,
        test_cfg=None,
        cache=False,
    ):

        super(CDDatasetStreet, self).__init__()
       # self.transform = Compose(transform)
        
        self.file_root = os.path.join(data_root,split)
        self.txt_path = os.path.join(data_root,split)
        self.txt_path = self.txt_path+".txt"
      

        self.voxel_size = voxel_size

        with open(self.txt_path, 'r') as f:
            self.list = f.readlines()
        self.file_size = len(self.list)
   
    def txt2sample(self,path):
    
        index = ['X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'label']
        with open(path, 'r') as f:
            lines = f.readlines()
            head = lines[0][2:].strip('\n').split(' ')
            ids = tuple([head.index(i) for i in index])
        points = np.loadtxt(path, skiprows=2, usecols = ids)   

        return points   

    
    def __getitem__(self,idx):
        
        p0_path =os.path.join(self.file_root, self.list[idx].split(' ')[0])
        p1_path = os.path.join(self.file_root, self.list[idx].split(' ')[1].strip())
       
        vertex0 = self.txt2sample(p0_path) 
        vertex1 = self.txt2sample(p1_path) 
        vertex0=vertex0.astype(np.float32)
        vertex1=vertex1.astype(np.float32)

        
        coord0=torch.from_numpy(vertex0[:,:3])
        coord1 = torch.from_numpy(vertex1[:,:3])
        rgb0=torch.from_numpy(vertex0[:,3:6])
        rgb1=torch.from_numpy(vertex1[:,3:6])
        label_cd_t0=torch.from_numpy(vertex0[:,6]).to(dtype=torch.long)
        label_cd_t1=torch.from_numpy(vertex1[:,6]).to(dtype=torch.long)
 
        min0 = torch.unsqueeze(coord0.min(0)[0], 0)

        coord0[:, 0] = (coord0[:, 0] - min0[0,0])  # x
        coord0[:, 1] = (coord0[:, 1] - min0[0,1])  # y
        coord0[:, 2] = (coord0[:, 2] - min0[0,2])  # z
        coord1[:, 0] = (coord1[:, 0] - min0[0,0])  # x
        coord1[:, 1] = (coord1[:, 1] - min0[0,1])  # y
        coord1[:, 2] = (coord1[:, 2] - min0[0,2])  # z
   
    
        if self.voxel_size>0:
            uniq_idx0 = voxelize(coord0.numpy(), self.voxel_size)
            coord0, rgb0,label_cd_t0= coord0[uniq_idx0], rgb0[uniq_idx0],label_cd_t0[uniq_idx0]
            uniq_idx1 = voxelize(coord1.numpy(), self.voxel_size)
            coord1, rgb1,label_cd_t1= coord1[uniq_idx1], rgb1[uniq_idx1],label_cd_t1[uniq_idx1]
 


        offset_t0=coord0.shape[0]
        offset_t1=coord1.shape[0]
        coord=torch.cat([coord0,coord1],dim=0)

        label_cd_t1[label_cd_t1==0]=2
        label_cd_t1[label_cd_t1==1]=3
  
        segment_cd=torch.cat([label_cd_t0,label_cd_t1],dim=0)

        ft0=torch.zeros(coord0.shape[0])   
        ft1=torch.ones(coord1.shape[0])   
        ft=torch.cat([ft0,ft1],dim=0).reshape(-1,1)
        rgb=torch.cat([rgb0,rgb1],dim=0)
       
        feat=torch.cat([coord,rgb],dim=1)
        featt=torch.cat([feat,ft],dim=1)
        
  
        data_dict = dict(
            coord=coord,
            feat=featt, #featt
            segment_cd=segment_cd,
            offset=coord.shape[0],
        )
     

        return data_dict

    def __len__(self):
      
        num=self.file_size
        return num
