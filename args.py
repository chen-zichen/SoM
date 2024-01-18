'''
args for test_seg.py
'''
import argparse
from semantic_sam.utils.arguments import load_opt_from_config_file
from seem.utils.distributed import init_distributed as init_distributed_seem



def get_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    
    parser.add_argument('--semsam_cfg', default="configs/semantic_sam_only_sa-1b_swinL.yaml", type=str)
    parser.add_argument('--seem_cfg', default="configs/seem_focall_unicl_lang_v1.yaml", type=str)
    parser.add_argument('--semsam_ckpt', default="./swinl_only_sam_many2many.pth", type=str)
    parser.add_argument('--sam_ckpt', default="./sam_vit_h_4b8939.pth", type=str)
    parser.add_argument('--seem_ckpt', default="./seem_focall_v1.pt", type=str)

    parser.add_argument('--device', default='cuda', type=str)
    return parser.parse_args()

def get_args_semsam():
    args = get_args()
    opt_semsam = load_opt_from_config_file(args.semsam_cfg)
    opt_seem = load_opt_from_config_file(args.seem_cfg)
    opt_seem = init_distributed_seem(opt_seem)
    return opt_semsam, opt_seem

