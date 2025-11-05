"""
MobileNetV2-based U-Net with ASPP and SE mechanisms.
"""
# Lazy import to avoid circular dependency issues
def get_mobilenet_model():
    """Get MobileNetV2ASPPResidualSEUNet model class"""
    from .mobinet_aspp_residual_se import MobileNetV2ASPPResidualSEUNet
    return MobileNetV2ASPPResidualSEUNet

__all__ = ['get_mobilenet_model']
