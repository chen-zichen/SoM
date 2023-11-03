# Set-of-Mark (SoM)
## demo_som.py
### Initialization and Configuration:

- load_opt_from_config_file(semsam_cfg): Loads options for the semsam model from a configuration file.
- load_opt_from_config_file(seem_cfg): Loads options for the seem model from a configuration file.
- init_distributed_seem(opt_seem): Initializes distributed computing settings for the seem model.

### Model Building and Loading:

- build_model(opt_semsam): Builds the semsam model based on the loaded options.
- build_model_seem(opt_seem): Builds the seem model based on the loaded options.
- BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda(): Instantiates the semsam model with pre-trained weights and prepares it for evaluation on a CUDA device.
- BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda(): Similar to the semsam model, it prepares the seem model for evaluation.

### Inference Functions:

- inference_semsam_m2m_auto: Performs inference with the semsam model in a many-to-many automatic setting.
- inference_sam_m2m_auto: Performs inference with the sam model in a many-to-many automatic setting.
- inference_sam_m2m_interactive: Performs interactive inference with the sam model.
- inference_seem_pano: Performs inference with the seem model for panoramic images.
- inference_seem_interactive: Performs interactive inference with the seem model.


### Custom Gradio Component:
- class ImageMask(gr.components.Image): A custom Gradio component for handling image inputs with sketching (masking) capabilities.

- runBtn.click(inference, ...): This line suggests that there is a Gradio button set up to trigger the inference function when clicked.
