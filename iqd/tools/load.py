from diffusers import DiffusionPipeline

from quantize.model.unet import QUNet2DModel


def load_trainer(model_path, quantizer):
    config, = DiffusionPipeline.load_config(model_path)

    scheduler = _load_component(config, 'scheduler', model_path)

    unet_path = tuple(config['unet'])
    if unet_path is ('diffusers', 'UNet2DModel'):
        unet = QUNet2DModel.convert_pretrained(model_path, quantizer)
    else:
        raise ValueError("Unsupported UNet to train.")

    return unet, scheduler
