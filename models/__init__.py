import logging
from models.classifier import Classifier, NormalizedClassifier
from models.img_resnet import ResNet50
from models.PM import PM
from models.Fusion import Fusion
from models.resnet101 import ResNet101
from models.resnet152 import ResNet152


def build_model50(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')
    # Build backbone
    logger.info("Initializing model: {}".format(config.MODEL.NAME))


    logger.info("Init model: '{}'".format(config.MODEL.NAME))
    model = ResNet50(config)

    model2 = PM(feature_dim=config.MODEL.FEATURE_DIM)
    fusion = Fusion(feature_dim=config.MODEL.FEATURE_DIM)
        
    logger.info("Model  size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    logger.info("Model2 size: {:.5f}M".format(sum(p.numel() for p in model2.parameters())/1000000.0))

    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    else:
        identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)

    clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)

    #classifier of new model
    clothes_classifier2 = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
    
    return model, model2, fusion, identity_classifier, clothes_classifier, clothes_classifier2

def build_model101(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')
    # Build backbone
    logger.info("Initializing model: {}".format(config.MODEL.NAME))


    logger.info("Init model: '{}'".format(config.MODEL.NAME))
    model = ResNet101(config)

    model2 = PM(feature_dim=config.MODEL.FEATURE_DIM)
    fusion = Fusion(feature_dim=config.MODEL.FEATURE_DIM)
        
    logger.info("Model  size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    logger.info("Model2 size: {:.5f}M".format(sum(p.numel() for p in model2.parameters())/1000000.0))

    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    else:
        identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)

    clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)

    #classifier of new model
    clothes_classifier2 = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
    
    return model, model2, fusion, identity_classifier, clothes_classifier, clothes_classifier2

def build_model152(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')
    # Build backbone
    logger.info("Initializing model: {}".format(config.MODEL.NAME))


    logger.info("Init model: '{}'".format(config.MODEL.NAME))
    model = ResNet152(config)

    model2 = PM(feature_dim=config.MODEL.FEATURE_DIM)
    fusion = Fusion(feature_dim=config.MODEL.FEATURE_DIM)
        
    logger.info("Model  size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    logger.info("Model2 size: {:.5f}M".format(sum(p.numel() for p in model2.parameters())/1000000.0))

    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    else:
        identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)

    clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)

    #classifier of new model
    clothes_classifier2 = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
    
    return model, model2, fusion, identity_classifier, clothes_classifier, clothes_classifier2