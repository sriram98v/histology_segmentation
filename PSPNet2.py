from typing import Optional, Union
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder



class PSPNet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: Optional[str] = 'imagenet', 
        encoder_depth: int = 3, 
        psp_out_channels: int = 512, 
        psp_use_batchnorm: bool = True, 
        psp_dropout: float = 0.2, 
        in_channels: int = 3, 
        classes: int = 1, 
        activation: Union[str, callable, None] = None, 
        upsampling: int = 8, 
        aux_params: Optional[dict] = None
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

