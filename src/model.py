import segmentation_models_pytorch as smp

def create_model(config):
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None, 
        in_channels=3,
        classes=1,
        activation=None 
    )
    return model
