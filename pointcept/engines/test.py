"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

from pointcept.datasets.cd_test import load_test_data,save_ply,save_ply_rgb
from  pointcept.utils.voxelize import voxelize



TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
#            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=False)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
#        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
      
        test_root=self.cfg.test_root
        subfolder=[f.path for f in os.scandir(test_root) if f.is_dir()]
        subfolders=[]
        names=[]
        for sub in subfolder:
            count=len([name for name in os.listdir(sub) if os.path.isfile(os.path.join(sub,name))])
            if count==2:
                subfolders.append(sub)
                name=os.path.basename(sub)
                names.append(name)

        record = {}

        for i in range(len(names)):

            data_name = str(i)
            end = time.time()
            filetest=os.path.join(self.cfg.test_root,names[i])
            input_dict,coord=load_test_data(filetest,self.cfg.test_voxel_size)

            if self.cfg.mask:
                input_dict["segment_cd"][input_dict["segment_cd"]>3]=-1
                input_dict["segment"][input_dict["segment"]>3]=-1

            segment = input_dict["segment_cd"].cpu().numpy()   

     
            pred = self.model(input_dict)["seg_logits"]  # (n, k)
          
          



            pred= F.softmax(pred, -1)
            pred = pred.max(1)[1].data.cpu().numpy()

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)
            
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
          

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    0,
                    i + 1,
                    0,
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

    
        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

      
            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class SemSegTester_Siam(TesterBase):
    def test(self):
#        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
      
        test_root=self.cfg.test_root
        subfolder=[f.path for f in os.scandir(test_root) if f.is_dir()]
        subfolders=[]
        names=[]
        for sub in subfolder:
            count=len([name for name in os.listdir(sub) if os.path.isfile(os.path.join(sub,name))])
            if count==2:
                subfolders.append(sub)
                name=os.path.basename(sub)
                names.append(name)

        record = {}
    
        for i in range(len(names)):#range(10):
            data_name = str(i)
            end = time.time()
            filetest=os.path.join(self.cfg.test_root,names[i])
            input_dict,coord=load_test_data_siam(filetest,self.cfg.test_voxel_size)
 


            if self.cfg.mask:
                input_dict["segment_cd"][input_dict["segment_cd"]>3]=-1
                input_dict["segment"][input_dict["segment"]>3]=-1

            



            pred = self.model(input_dict)["seg_logits"]  # (n, k)
           
            segment = input_dict["segment_cd"].cpu().numpy()
            pred= F.softmax(pred, -1)
            pred = pred.max(1)[1].data.cpu().numpy()
            #save test result
            fname,extension=os.path.splitext(os.path.basename(names[i]))
            num=segment.shape[0]-coord.shape[0]
            # print(filetest)
            # print(torch.from_numpy(pred).unique(return_counts=True))
            #save_ply(coord, pred[num:],segment[num:],self.cfg.save_file, fname)
         
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)
            
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
          

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    0,
                    i + 1,
                    0,
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

      
            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")


@TESTERS.register_module()
class SemSegBiTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
      

        self.txt_path=self.cfg.test_files
        self.file_root=self.cfg.test_root
        with open(self.txt_path, 'r') as f:
            self.list = f.readlines()
        self.file_size = len(self.list)
        record = {}
     
        for i in range(self.file_size ):

            data_name = str(i)
            end = time.time()
            input_dict,c0_num,coord_ori,rgb0,rgb1=self.load_data(i)
 
            segment = input_dict["segment_cd"].cpu().numpy()
          
      
            pred = self.model(input_dict)["seg_logits"]  # (n, k)
            pred= F.softmax(pred, -1)
            pred = pred.max(1)[1].data.cpu().numpy()
            #save test result
#            fname,extension=os.path.splitext(os.path.basename(names[i]))
            #num=segment.shape[0]-coord.shape[0]
            #save_ply(coord, segment[num:],self.cfg.save_file, fname)
            
            pred[pred==2]=0
            segment[segment==2]=0
            pred[pred==3]=2
            segment[segment==3]=2
            n0=str(i)+"_0"
            n1=str(i)+"_1"
          
    
            rgb0=rgb0*256
            rgb1=rgb1*256
            save_ply_rgb(coord_ori[:c0_num],rgb0, pred[:c0_num],segment[:c0_num],self.cfg.save_file, n0)
            save_ply_rgb(coord_ori[c0_num:], rgb1,pred[c0_num:],segment[c0_num:],self.cfg.save_file, n1)
            print(i,torch.tensor(segment).bincount())
            
            
            
            intersection, union, target = intersection_and_union(
                pred, segment, 3, self.cfg.data.ignore_index
            )
          
            # intersection, union, target = intersection_and_union(
            #     pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            # )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)
           
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            # print(mask,iou_class)
         

        
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    0,
                    i + 1,
                    0,
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
           

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

      
            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(3):#self.cfg.data.num_classes-1
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
    
    def txt2sample(self,path):
    
        index = ['X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'label']
        with open(path, 'r') as f:
            lines = f.readlines()
            head = lines[0][2:].strip('\n').split(' ')
            ids = tuple([head.index(i) for i in index])
        points = np.loadtxt(path, skiprows=2, usecols = ids)   

        return points
   
    def load_data(self,idx):
        
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
   
        self.voxel_size=0.03
        if self.voxel_size>0:
            uniq_idx0 = voxelize(coord0.numpy(), self.voxel_size)
            coord0, rgb0,label_cd_t0= coord0[uniq_idx0], rgb0[uniq_idx0],label_cd_t0[uniq_idx0]
            uniq_idx1 = voxelize(coord1.numpy(), self.voxel_size)
            coord1, rgb1,label_cd_t1= coord1[uniq_idx1], rgb1[uniq_idx1],label_cd_t1[uniq_idx1]
 


        offset_t0=coord0.shape[0]
        offset_t1=coord1.shape[0]
        coord=torch.cat([coord0,coord1],dim=0)
        coord_ori=coord.clone()
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
            coord=coord.cuda(),
            feat=featt.cuda(), #featt
            #segment=rgb.long(),
            segment_cd=segment_cd.cuda(),
            batch=torch.zeros(coord.shape[0]).int().cuda(),

        )
        coord_ori[:, 0] = (coord_ori[:, 0] + min0[0,0])  # x
        coord_ori[:, 1] = (coord_ori[:, 1] + min0[0,1])  # y
        coord_ori[:, 2] = (coord_ori[:, 2] + min0[0,2])  # z

        return data_dict, coord0.shape[0],coord_ori,rgb0,rgb1