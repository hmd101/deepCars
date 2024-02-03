# Transfer Learning to Predict Modernity of Cars from Images

In this project, we finetune a pre-trained ResNet to classify cars according to their production year and subsequently compute a score for their modernity. We also investigate which pixels in an image are most informative for the resulting prediction.

## Finetuning Pre-Trained Model

We use a [pretrained ResNet-18](./src/model_weights.py) available from `torchvision` for which we freeze its trainable parameters and add an additional trainable `nn.Linear` layer used for finetuning:

https://github.com/hmd101/deepCars/blob/edefb962eaa136e739749cc91c945514f0e5a63a/src/models.py#L5-L33

## Training

We [trained](src/training.py) the model on the CPU of a MacBook Pro with 32GB of RAM. This places restrictions on the possible size of the training set and the number of epochs.
Below we show the result of a training run on 10000 training datapoints run for 10 epochs. Both train and test loss reduce, while the accuracy increases, even if given the computational constraints predictive performance is still limited. 

<div align="center">
    <img src="results/debug/nyu/model-performance/learning_curves.png" width=600>
</div>

## Modernity Score

A car's [modernity](src/metrics.py) is scored according to the probabilities for different production year categories.

<div align="center">
    <img src="results/debug/nyu/modernity/modernity_score_per_year.png" width=600>
</div>

## Model Evaluation

### Class Imbalance
One reason for the decrease in test cross-entropy loss, but limited accuracy after an hour of training could be the stark class imbalance. Many more cars from more recent years exist in the dataset:

<div align="center">
    <img src="results/debug/nyu/dataset-statistics/Statistics_Train Dataset.png" width=600>
</div>

### Confusion Matrix
Taking a look at the confusion matrix shows that the decrease in cross-entropy loss on the test set corresponds to the model learning to predict rough year ranges for cars, that were produced more recently, even if the exact production year is not yet correctly classified.

<div align="center">
    <img src="results/debug/nyu/model_interpretability/confusion_matrix.png" width=600>
</div>

## Saliency Map
We can understand what pixels in input space are most informative by looking at a [saliency map](src/analysis/saliency_map.ipynb). It highlights pixels according to gradient magnitude for the predicted class score. 

<div align="center">
    <img src="results/debug/nyu/model_interpretability/saliency_map.png" width=600>
</div>