import models.resnet as resnet
import models.vgg as vgg
import models.cyresnet as cyresnet
import models.cyvgg as cyvgg

def get_model(model, dataset, classify=True):

    """
    VGG Models
    """
    if model == 'vgg11':
        model = vgg.vgg11_bn(dataset=dataset, classify=classify)
    if model == 'vgg13':
        model = vgg.vgg13_bn(dataset=dataset, classify=classify)
    if model == 'vgg16':
        model = vgg.vgg16_bn(dataset=dataset, classify=classify)
    if model == 'vgg19':
        model = vgg.vgg19_bn(dataset=dataset, classify=classify)

    """
    CyVGG Models
    """
    if model == 'cyvgg11':
        model = cyvgg.cyvgg11_bn(dataset=dataset, classify=classify)
    if model == 'cyvgg13':
        model = cyvgg.cyvgg13_bn(dataset=dataset, classify=classify)
    if model == 'cyvgg16':
        model = cyvgg.cyvgg16_bn(dataset=dataset, classify=classify)
    if model == 'cyvgg19':
        model = cyvgg.cyvgg19_bn(dataset=dataset, classify=classify)

    """
    Resnet Models   
    """
    if model == 'resnet20':
        model = resnet.resnet20(dataset=dataset)
    if model == 'resnet32':
        model = resnet.resnet32(dataset=dataset)
    if model == 'resnet44':
        model = resnet.resnet44(dataset=dataset)
    if model == 'resnet56':
        model = resnet.resnet56(dataset=dataset)

    """
    CyResnet Models
    """
    if model == 'cyresnet20':
        model = cyresnet.cyresnet20(dataset=dataset)
    if model == 'cyresnet32':
        model = cyresnet.cyresnet32(dataset=dataset)
    if model == 'cyresnet44':
        model = cyresnet.cyresnet44(dataset=dataset)
    if model == 'cyresnet56':
        model = cyresnet.cyresnet56(dataset=dataset)

    return model

