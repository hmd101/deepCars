import torchvision.models as models
import torch


def get_pretrained_model():
    return models.resnet18(weights="IMAGENET1K_V1")


def get_fine_tuneable_model(
    num_classes: int,
    pretrained_model: torch.nn.Module = get_pretrained_model(),
):
    in_features_final_layer = list(pretrained_model.parameters())[-2].size()[
        1
    ]  # in-features to final layer of resnet18

    finetuneable_model = torch.nn.Sequential(
        *(list(pretrained_model.children())[:-1]), torch.nn.Flatten()
    )  # add all but final layer from pretrained model

    # freeze paramaters of pretrained model:
    for param in finetuneable_model.parameters():
        param.requires_grad = False

    # add custom final layer to resnet18
    finetuneable_model.add_module(
        "fc",
        torch.nn.Linear(in_features=in_features_final_layer, out_features=num_classes),
    )

    # # apply softmax to final layers' outputs
    # finetuneable_model.add_module("sm", torch.nn.Softmax(dim=0))
    return finetuneable_model


def get_fine_tuned_model(num_classes: int, path_to_weights: str):
    """Creates a fine-tuned model by loading weights from file."""
    model = get_fine_tuneable_model(num_classes=num_classes)
    model.load_state_dict(torch.load(path_to_weights))
    return model
