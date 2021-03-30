from .triplet import ContrastiveLoss, TripletLoss, TripletNet
from .frame import FRAME_MODELS, Resnet50_RMAC, MobileNet_AVG, Resnet50_avgpool_FC3, Resnet50_ccpool_FC3
from .segment import SEGMENT_MODELS, Segment_MaxPool, Segment_AvgPool


def get_frame_model(name, **kwargs):
    assert name in FRAME_MODELS
    return getattr(frame, name)()


def get_segment_model(name, **kwargs):
    assert name in SEGMENT_MODELS
    return getattr(segment, name)()
