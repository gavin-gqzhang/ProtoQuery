# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.abspath(os.path.join(current_dir,'../')))

import argparse
import time
import torch

from tools.relation_train_net import fix_eval_modules
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.comm import synchronize
import numpy as np
import random
from matplotlib import pyplot as plt

def load_reps():
    sample_ids=os.listdir("/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/features/")
    dpplml_path,pe_path='/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/features/','/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/pe_features'
    
    pe_sub_emb,pe_obj_emb,pe_rel_reps,pe_rel_labels=[],[],[],[]
    
    dpplml_sub_emb,dpplml_obj_emb,dpplml_entity_rel_rep=[],[],[]
    dpplml_s_p_rep,dpplml_o_p_rep,dpplml_rel_rep,dpplml_rel_center,dpplml_rel_labels=[],[],[],[],[]
    # for sample_id in tqdm(random.sample(sample_ids,k=1000)):
    for sample_id in tqdm(sample_ids):
        # ************ *************** ************
        dpplml_entity_dicts=torch.load(f'{dpplml_path}/{sample_id}/entity_reps.pth',map_location='cpu')
        dpplml_rel_dicts=torch.load(f'{dpplml_path}/{sample_id}/rel_reps.pth',map_location='cpu')
        
        dpplml_labels=dpplml_rel_dicts['rel_label'].cpu()
        
        # ************ PE Net Features ************
        pe_entity_dicts=torch.load(f'{pe_path}/{sample_id}/entity_features.pth',map_location='cpu')
        pe_rel_dicts=torch.load(f'{pe_path}/{sample_id}/rel_features.pth',map_location='cpu')
        
        pe_labels=pe_rel_dicts['rel_labels'].cpu()
        
        # ************ *************** ************
        # continue_epoch=False
        assert len(pe_labels)==len(dpplml_labels)
        # for pe_l,dpplml_l in zip(pe_rel_dicts['rel_labels'].cpu(),dpplml_labels[fg_mask]):
        #     if not torch.equal(pe_l,dpplml_l):
        #         print(f'PE Net labels: {pe_rel_dicts["rel_labels"]}, DPPLML labels: {dpplml_labels[fg_mask]}')
        #         continue_epoch=True
        #         break
        
        # if continue_epoch:
        #     continue
        
        # ************ insert data ************
        
        dpplml_sub_emb.append(dpplml_entity_dicts['sub_embeds'].cpu())
        dpplml_obj_emb.append(dpplml_entity_dicts['obj_embeds'].cpu())
        dpplml_entity_rel_rep.append(dpplml_entity_dicts['rel_reps'])
        
        dpplml_s_p_rep.append(dpplml_rel_dicts['sp_query'].cpu())
        dpplml_o_p_rep.append(dpplml_rel_dicts['op_query'].cpu())
        dpplml_rel_rep.append(dpplml_rel_dicts['rel_query'].cpu())
        dpplml_rel_center.append(dpplml_rel_dicts['rel_center'].cpu())  # all_samples,num_rels,hidden_dim
        dpplml_pro=dpplml_rel_dicts['predicate_prototye'].cpu() # num_rels,hidden_dim
        dpplml_rel_labels.append(dpplml_labels)
        
        pe_sub_emb.append(pe_entity_dicts['sub_sem_reps'].cpu())
        pe_obj_emb.append(pe_entity_dicts['obj_sem_reps'].cpu())
        pe_pro=pe_rel_dicts['predicate_proto'].cpu()
        
        pe_rel_reps.append(pe_rel_dicts['rel_sem_reps'].cpu())
        pe_rel_labels.append(pe_labels)
        
        null_cls=False
        for rel_id in range(1,51):
            exist_labels=torch.cat(dpplml_rel_labels,dim=0)
            if torch.sum(exist_labels==rel_id)<30:
                null_cls=True
                break
        
        if not null_cls:
            print(f'Each class have features more than fifty.')
            break
             
    
    return torch.cat(dpplml_sub_emb,dim=0),torch.cat(dpplml_obj_emb,dim=0),torch.cat(dpplml_entity_rel_rep,dim=0),torch.cat(dpplml_s_p_rep,dim=0),torch.cat(dpplml_o_p_rep,dim=0),torch.cat(dpplml_rel_rep,dim=0),torch.cat(dpplml_rel_center,dim=0),dpplml_pro,torch.cat(dpplml_rel_labels,dim=0),torch.cat(pe_sub_emb,dim=0),torch.cat(pe_obj_emb,dim=0),torch.cat(pe_rel_reps,dim=0),pe_pro,torch.cat(pe_rel_labels,dim=0)

def intra_inter_var(reps,rel_pros,labels,cls_num=51):
    wcv,bcv=[],[]
    overall_center = torch.mean(reps, dim=0)
    for i in range(1,cls_num):
        cls_points=reps[labels==i]
        cls_center = rel_pros[i]
        wcv.append(torch.sum((cls_points - cls_center) ** 2).item()/len(cls_points))
        
        bcv.append(torch.sum(labels==i) * torch.sum((torch.mean(cls_points,dim=0) - overall_center) ** 2).item())
        
    return wcv,bcv

def intra_inter_fea_var(sub_reps,rel_reps):
    return (torch.sum(sub_reps-rel_reps,dim=-1)**2)

    
if __name__ == "__main__":
    if not os.path.exists("/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/load_reps.pth"):
        dpplml_sub_emb,dpplml_obj_emb,dpplml_entity_rel,dpplml_s_p_rep,dpplml_o_p_rep,dpplml_rel_rep,dpplml_rel_center,dpplml_rel_pro,dpplml_rel_label,pe_sub_emb,pe_obj_emb,pe_rel_rep,pe_rel_pro,pe_rel_label=load_reps()
        save_load_features=dict(dpplml_sub_emb=dpplml_sub_emb,dpplml_obj_emb=dpplml_obj_emb,dpplml_entity_rel=dpplml_entity_rel,dpplml_s_p_rep=dpplml_s_p_rep,dpplml_o_p_rep=dpplml_o_p_rep,dpplml_rel_rep=dpplml_rel_rep,dpplml_rel_center=dpplml_rel_center,dpplml_rel_pro=dpplml_rel_pro,dpplml_rel_label=dpplml_rel_label,pe_sub_emb=pe_sub_emb,pe_obj_emb=pe_obj_emb,pe_rel_rep=pe_rel_rep,pe_rel_pro=pe_rel_pro,pe_rel_label=pe_rel_label)
        torch.save(save_load_features,"/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/load_reps.pth")
    else:
        load_features=torch.load("/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/load_reps.pth",map_location='cpu')
        dpplml_sub_emb,dpplml_obj_emb,dpplml_entity_rel,dpplml_s_p_rep,dpplml_o_p_rep,dpplml_rel_rep,dpplml_rel_center,dpplml_rel_pro,dpplml_rel_label,pe_sub_emb,pe_obj_emb,pe_rel_rep,pe_rel_pro,pe_rel_label=load_features['dpplml_sub_emb'],load_features['dpplml_obj_emb'],load_features['dpplml_entity_rel'],load_features['dpplml_s_p_rep'],load_features['dpplml_o_p_rep'],load_features['dpplml_rel_rep'],load_features['dpplml_rel_center'],load_features['dpplml_rel_pro'],load_features['dpplml_rel_label'],load_features['pe_sub_emb'],load_features['pe_obj_emb'],load_features['pe_rel_rep'],load_features['pe_rel_pro'],load_features['pe_rel_label']

    dpplml_rel_nums,pe_rel_nums=[],[]

    save_path='vis_res'
    os.makedirs(save_path,exist_ok=True)

    assert dpplml_sub_emb.shape[0]==dpplml_obj_emb.shape[0]==dpplml_entity_rel.shape[0]==dpplml_s_p_rep.shape[0]==dpplml_o_p_rep.shape[0]==dpplml_rel_rep.shape[0]==dpplml_rel_label.shape[0]==pe_sub_emb.shape[0]==pe_obj_emb.shape[0]==pe_rel_rep.shape[0]==pe_rel_label.shape[0]
    sample_nums=dpplml_sub_emb.shape[0]
    sample_choice_idx=random.sample(range(sample_nums),k=1000)

    # *************** within and between cls variance ***************

    vocab_file = json.load(open('/data/sdc/SGG_data/VG/VG-SGG-dicts.json'))
    idx2pred = vocab_file['idx_to_predicate']

    pe_wcv,_=intra_inter_var(pe_rel_rep,pe_rel_pro,pe_rel_label)
    dpplml_wcv,_=intra_inter_var(dpplml_rel_rep,dpplml_rel_pro,dpplml_rel_label)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1,51),pe_wcv,color="blue",label='PE variance')
    plt.plot(range(1,51),dpplml_wcv,color="red",label='ProtoQuery variance')

    plt.xticks([])
    plt.xlabel('Predicate Classes')
    plt.ylabel('Variance of Similar Predicate Features and Predicate Prototypes')

    plt.legend()
    plt.savefig(f'{save_path}/intra_cls_var.png')
    plt.clf()

    pe_sub_rel_wcv=intra_inter_fea_var(pe_sub_emb,pe_rel_rep)
    pe_obj_rel_wcv=intra_inter_fea_var(pe_obj_emb,pe_rel_rep)

    dpplml_sub_rel_wcv=intra_inter_fea_var(dpplml_sub_emb,dpplml_rel_rep)
    dpplml_obj_rel_wcv=intra_inter_fea_var(dpplml_obj_emb,dpplml_rel_rep)

    max_pe_sub_var,max_pe_obj_var=torch.max(pe_sub_rel_wcv),torch.max(pe_obj_rel_wcv)
    max_sample_sub_id,max_sample_obj_id=pe_sub_rel_wcv==max_pe_sub_var,pe_obj_rel_wcv==max_pe_obj_var

    pe_sub_rel_wcv=pe_sub_rel_wcv[max_sample_sub_id]
    pe_obj_rel_wcv=pe_obj_rel_wcv[max_sample_obj_id]
    dpplml_sub_rel_wcv=dpplml_sub_rel_wcv[max_sample_sub_id]
    dpplml_obj_rel_wcv=dpplml_obj_rel_wcv[max_sample_obj_id]

    sub_sample_idx=random.sample(range(pe_sub_rel_wcv.shape[0]),k=1000)
    obj_sample_idx=random.sample(range(pe_obj_rel_wcv.shape[0]),k=1000)

    plt.plot(range(1000),pe_sub_rel_wcv[sub_sample_idx],color="blue",label='PE variance')
    plt.plot(range(1000),dpplml_sub_rel_wcv[sub_sample_idx],color="red",label='ProtoQuery variance')

    plt.xlabel('Samples')
    plt.ylabel('Variance of Subject and Predicate Features Within the Same Sample')

    plt.legend()
    plt.savefig(f'{save_path}/sub_rel_var.png')

    plt.clf()

    plt.plot(range(1000),pe_obj_rel_wcv[obj_sample_idx],color="blue",label='PE variance')
    plt.plot(range(1000),dpplml_obj_rel_wcv[obj_sample_idx],color="red",label='ProtoQuery variance')

    plt.xlabel('Samples')
    plt.ylabel('Variance of Object and Predicate Features Within the Same Sample')

    plt.legend()
    plt.savefig(f'{save_path}/obj_rel_var.png')
