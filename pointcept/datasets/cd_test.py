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
import laspy

def load_test_data(folderpath,test_voxel_size=0.5):

    curDir = os.scandir(folderpath)
    for f in curDir:
        
        if f.name == "pointCloud0.ply":
            filesPC0=f.path
        elif f.name == "pointCloud1.ply":
            filesPC1=f.path

        
 

    vertex0=read_from_ply_PC0(filesPC0)
    coord0=torch.from_numpy(vertex0[:,:3])


    vertex1=read_from_ply_PC1(filesPC1)
 
    coord1 = torch.from_numpy(vertex1[:,:3])
    label_cd=torch.from_numpy(vertex1[:,3]).to(dtype=torch.long)
    label_seg_t1=torch.from_numpy(vertex1[:,4]).to(dtype=torch.long)
    label_seg_t0=torch.from_numpy(vertex0[:,3]).to(dtype=torch.long)
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

    voxel_size=test_voxel_size

    if voxel_size>0:
        uniq_idx0 = voxelize(coord0.numpy(), voxel_size)
        coord0, feat0,label_seg_t0= coord0[uniq_idx0], feat0[uniq_idx0],label_seg_t0[uniq_idx0]
        uniq_idx1 = voxelize(coord1.numpy(), voxel_size)
        coord1, feat1,label_cd,label_seg_t1= coord1[uniq_idx1], feat1[uniq_idx1],label_cd[uniq_idx1],label_seg_t1[uniq_idx1]

    coord=torch.cat([coord0,coord1],dim=0)

    
    coord=torch.cat([coord0,coord1],dim=0)
    seg_cd0=torch.full([coord0.shape[0]],-1)
    seg_cd=torch.cat([seg_cd0,label_cd],dim=0)
    segment=torch.cat([label_seg_t0,label_seg_t1],dim=0)

    ft0=torch.zeros(coord0.shape[0])   
    ft1=torch.ones(coord1.shape[0])   
    ft=torch.cat([ft0,ft1],dim=0).reshape(-1,1)
       
    feat=torch.cat([feat0,feat1],dim=0)
        
    featt=torch.cat([coord,ft],dim=1)
    
    
    #SEG=3
    # mask=segment>2
    # segment[mask]=2
    data_dict = dict(
        coord=coord.cuda(),
        feat=featt.cuda(), #featt
        segment_cd=seg_cd.cuda(),
        segment=segment.cuda(),
        #offset=torch.tensor(coord.shape[0]),
        batch=torch.zeros(coord.shape[0]).int().cuda(),
    ) 
    
    coord1[:, 0] = (coord1[:, 0] + min0[0,0])  # x
    coord1[:, 1] = (coord1[:, 1] + min0[0,1])  # y
    coord1[:, 2] = (coord1[:, 2] + min0[0,2]) 
    return data_dict,coord1

def load_test_data_siam(folderpath,test_voxel_size=0.5):

    curDir = os.scandir(folderpath)
    for f in curDir:
        
        if f.name == "pointCloud0.ply":
            filesPC0=f.path
        elif f.name == "pointCloud1.ply":
            filesPC1=f.path

        
 

    vertex0=read_from_ply_PC0(filesPC0)
    coord0=torch.from_numpy(vertex0[:,:3])


    vertex1=read_from_ply_PC1(filesPC1)
 
    coord1 = torch.from_numpy(vertex1[:,:3])
    label_cd=torch.from_numpy(vertex1[:,3]).to(dtype=torch.long)
    label_seg_t1=torch.from_numpy(vertex1[:,4]).to(dtype=torch.long)
    label_seg_t0=torch.from_numpy(vertex0[:,3]).to(dtype=torch.long)
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

    voxel_size=test_voxel_size

    if voxel_size>0:
        uniq_idx0 = voxelize(coord0.numpy(), voxel_size)
        coord0, feat0,label_seg_t0= coord0[uniq_idx0], feat0[uniq_idx0],label_seg_t0[uniq_idx0]
        uniq_idx1 = voxelize(coord1.numpy(), voxel_size)
        coord1, feat1,label_cd,label_seg_t1= coord1[uniq_idx1], feat1[uniq_idx1],label_cd[uniq_idx1],label_seg_t1[uniq_idx1]

    coord=torch.cat([coord0,coord1],dim=0)

    
    coord=torch.cat([coord0,coord1],dim=0)
    seg_cd0=torch.full([coord0.shape[0]],-1)
    seg_cd=torch.cat([seg_cd0,label_cd],dim=0)
    segment=torch.cat([label_seg_t0,label_seg_t1],dim=0)

    ft0=torch.zeros(coord0.shape[0])   
    ft1=torch.ones(coord1.shape[0])   
    ft=torch.cat([ft0,ft1],dim=0).reshape(-1,1)
       
    feat=torch.cat([feat0,feat1],dim=0)
        
    featt=torch.cat([coord,ft],dim=1)
    
    
    #SEG=3
    # mask=segment>2
    # segment[mask]=2
  
    
    data_dict = dict(
            coord0=coord0.cuda(),
            feat0=coord0.cuda(), 
            coord1=coord1.cuda(),
            feat1=coord1.cuda(),
            batch_t0=torch.zeros(coord0.shape[0]).int().cuda(),
            batch_t1=torch.zeros(coord1.shape[0]).int().cuda(),
            segment_cd_t0=seg_cd0.cuda(),
            segment_cd_t1=label_cd.cuda(),
            segment_t0=label_seg_t0.cuda(),
            segment_t1=label_seg_t1.cuda(),
        )
    coord1[:, 0] = (coord1[:, 0] + min0[0,0])  # x
    coord1[:, 1] = (coord1[:, 1] + min0[0,1])  # y
    coord1[:, 2] = (coord1[:, 2] + min0[0,2]) 
    return data_dict,coord1


def read_from_ply_PC0(filename, nameInPly="vertex"):
        """read XYZ for each vertex."""
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

def read_from_ply_PC1(filename, nameInPly="vertex"):
        """read XYZ for each vertex."""
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

def load_test_data_shrec(folderpath):
        
        vertices0,vertices1=read_from_ply_XYZRGB(folderpath)
    
        label_ch=vertices1[0,6]
     
        coord0=vertices0[:,:3]
        coord1=vertices1[:,:3]
        feat0=vertices0[:,3:6]
        feat1=vertices1[:,3:6]
    
        #normalize
        min0 = torch.unsqueeze(coord0.min(0)[0], 0)
        coord0[:, 0] = (coord0[:, 0] - min0[0,0])  # x
        coord0[:, 1] = (coord0[:, 1] - min0[0,1])  # y
        coord0[:, 2] = (coord0[:, 2] - min0[0,2])  # z
        coord1[:, 0] = (coord1[:, 0] - min0[0,0])  # x
        coord1[:, 1] = (coord1[:, 1] - min0[0,1])  # y
        coord1[:, 2] = (coord1[:, 2] - min0[0,2])  # z
       
     
        voxel_size=0.06

        if voxel_size>0:
            uniq_idx0 = voxelize(coord0.numpy(), voxel_size)
            coord0, feat0= coord0[uniq_idx0], feat0[uniq_idx0]
            uniq_idx1 = voxelize(coord1.numpy(), voxel_size)
            coord1, feat1= coord1[uniq_idx1], feat1[uniq_idx1]
   
        coord=torch.cat([coord0,coord1],dim=0)
        ft0=torch.zeros(coord0.shape[0])   
        ft1=torch.ones(coord1.shape[0])   
        ft=torch.cat([ft0,ft1],dim=0).reshape(-1,1)
        feat=torch.cat([feat0,feat1],dim=0)  
        featt=torch.cat([coord,feat,ft],dim=1)
    
        category=torch.ones(1)*label_ch.long()
        batch=torch.LongTensor(coord.shape[0])*0
        data_dict = dict(
            coord=coord.cuda(),
            feat=featt.cuda(), #featt
            category=category.cuda(),
            batch=batch.cuda(),
            pt_num=torch.tensor([coord0.shape[0],coord1.shape[0]]).cuda(),

        )  

        return data_dict



def save_las(coord, label,folder, fname):
    print("saving")
    format = ".las"
    path_out = os.path.join( folder, fname + format)
    new_hdr = laspy.LasHeader(version="1.2", point_format=3)
    new_hdr.scales = [0.01, 0.01, 0.01]
    pred_las = laspy.LasData(new_hdr)
    x = coord[:,0]
    y = coord[:,1]
    z = coord[:,2]
    ll= label
    pred_las.x = x#my_data[:,0]
    pred_las.y = y#my_data[:,1]
    pred_las.z = z#my_data[:,2]
    pred_las.classification = ll#my_data[:,3]
    pred_las.write(path_out)

def load_las(path):
    input_las = laspy.read(path)
    point_records = input_las.points.copy()
    las_scaleX = input_las.header.scale[0]
    las_offsetX = input_las.header.offset[0]
    las_scaleY = input_las.header.scale[1]
    las_offsetY = input_las.header.offset[1]
    las_scaleZ = input_las.header.scale[2]
    las_offsetZ = input_las.header.offset[2]

    # calculating coordinates
    p_X = np.array((point_records['X'] * las_scaleX) + las_offsetX)
    p_Y = np.array((point_records['Y'] * las_scaleY) + las_offsetY)
    p_Z = np.array((point_records['Z'] * las_scaleZ) + las_offsetZ)
    p_lable=np.array((point_records["classification"] ))
    points = np.vstack((p_X,p_Y,p_Z,p_lable)).T
    
    return points
def save_ply(coord,pred, gt,folder, fname):
    format = ".ply"
    path_out = os.path.join( folder, fname + format)
    pointsdata=[(coord[i,0],coord[i,1],coord[i,2],pred[i],gt[i]) for i in range(coord.shape[0])]
    vertex=np.array(pointsdata,dtype=[('x','f8'),('y','f8'),('z','f8'),('pred','i4'),('gt','i4')])
    el=PlyElement.describe(vertex,'vertex') 
    plydata=PlyData([el])
    plydata.write(path_out)   

def save_ply_rgb(coord,rgb,pred, gt,folder, fname):
    format = ".ply"
    path_out = os.path.join( folder, fname + format)
    pointsdata=[(coord[i,0],coord[i,1],coord[i,2],rgb[i,0],rgb[i,1],rgb[i,2],pred[i],gt[i]) for i in range(coord.shape[0])]
    vertex=np.array(pointsdata,dtype=[('x','f8'),('y','f8'),('z','f8'),('red','i4'),('green','i4'),('blue','i4'),('pred','i4'),('gt','i4')])
    el=PlyElement.describe(vertex,'vertex') 
    plydata=PlyData([el])
    plydata.write(path_out)   

def read_from_ply_XYZRGB(folderpath, nameInPly="vertex"):
        """read XYZ for each vertex."""
        filename0=folderpath+"/pointCloud0.ply"
        filename1=folderpath+"/pointCloud1.ply"

        #print(filename1)
        with open(filename0, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices0 = np.zeros(shape=[num_verts, 7], dtype=np.float32)
            vertices0[:, 0] = plydata[nameInPly].data["x"]
            vertices0[:, 1] = plydata[nameInPly].data["y"]
            vertices0[:, 2] = plydata[nameInPly].data["z"]
            vertices0[:, 3] = plydata[nameInPly].data["r"]
            vertices0[:, 4] = plydata[nameInPly].data["g"]
            vertices0[:, 5] = plydata[nameInPly].data["b"]
            vertices0[:, 6] = plydata[nameInPly].data["classification"]
        with open(filename1, "rb") as f:
            plydata = PlyData.read(f)
            num_verts = plydata[nameInPly].count
            vertices1 = np.zeros(shape=[num_verts, 7], dtype=np.float32)
            vertices1[:, 0] = plydata[nameInPly].data["x"]
            vertices1[:, 1] = plydata[nameInPly].data["y"]
            vertices1[:, 2] = plydata[nameInPly].data["z"]
            vertices1[:, 3] = plydata[nameInPly].data["r"]
            vertices1[:, 4] = plydata[nameInPly].data["g"]
            vertices1[:, 5] = plydata[nameInPly].data["b"]
            vertices1[:, 6] = plydata[nameInPly].data["classification"]
      
        return torch.from_numpy(vertices0),torch.from_numpy(vertices1)

