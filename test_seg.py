'''
This is a file for incorporating segmentation models into the multimodal explanation framework.
'''

import torch
from PIL import Image

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from scipy.ndimage import label
import numpy as np

# import args
import args


class SegmentationModel():
    def __init__(self, images, alpha, label_mode, anno_mode, *args, **kwargs):

        self.images = images
        self.alpha = alpha
        self.label_mode = label_mode
        self.anno_mode = anno_mode

    def args(self, ):
        # initialize args
        get_args = args.get_args()
        seem_ckpt = get_args.seem_ckpt
        opt_semsam, opt_seem = args.get_args_semsam()
        return opt_semsam, opt_seem, seem_ckpt

    def inference(self, ):
        if not isinstance(self.images, list):
            self.images = [self.images]

        # model: seem, sam,   semantic-sam(level:1-6)
        opt_seem, opt_seem, seem_ckpt = self.args()
        model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
        model = model_seem

        text_size = 640

        # inference
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # anno_mode: "Mask", "Box", "Mark"
            outputs = []
            for image in self.images:
                output,_ = inference_seem_pano(model, image, text_size, self.label_mode, self.alpha, self.anno_mode)
                outputs.append(output)
            return outputs



if __name__ == '__main__':

    # inference
    image = Image.open('examples/output.png')
    print("start")

    output = SegmentationModel(image, alpha=0.2, label_mode='1', anno_mode=['Mark', 'Mask'])
    output = output.inference()
    

    # save image
    for i, out in enumerate(output):
        output_image = Image.fromarray(out)
        output_image.save(f'examples/output_{i}.png')


