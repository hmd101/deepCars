import torchvision.models as models
import torch 

pretrained_model = models.resnet18(pretrained=True)

def get_fine_tuneable_model(num_classes:int, pretrained_model:torch.nn.Module = models.resnet18(pretrained=True)):

    in_features_final_layer = list(pretrained_model.parameters())[-2].size()[1]# in-features to final layer of resnet18

    finetuneable_model = torch.nn.Sequential(*(list(pretrained_model.children())[:-1]), torch.nn.Flatten()) # add all but final layer from pretrained model

    # freeze paramaters of pretrained model:
    for param in finetuneable_model.parameters():
        param.requires_grad = False

    # add custom final layer to resnet18
    finetuneable_model.add_module(
        "fc",
        torch.nn.Linear(
            in_features = in_features_final_layer,
            out_features = num_classes
        ),
    )

    # apply softmax to final layers' outputs
    finetuneable_model.add_module("sm", torch.nn.Softmax(dim=0))
    return finetuneable_model