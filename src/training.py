import fire
import torch
from typing import Optional
import time
import pandas as pd

from data.car_dataset import (
    CarDataset,
    data_transforms,
    inverse_transform,
    year2label_fn,
)

from data.data_splitting import split_dataset_dfs
from data import data_cleaning
from models import get_fine_tuneable_model
from tqdm import tqdm
import datetime


def training_loop(
    model,
    loss_fn,
    optimizer,
    data_loader_dict,
    num_epochs=3,
    lr_scheduler=None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    # Training and test metrics
    num_batches = 0
    batch_size = len(data_loader_dict["train"])
    start_time = time.time()
    train_loss_batches_list = []
    train_accuracy_batches_list = []
    train_loss_epochs_list = []
    train_accuracy_epochs_list = []
    train_time_epochs_list = []
    test_loss_epochs_list = []
    test_accuracy_epochs_list = []

    for epoch in range(num_epochs):
        # Initialize training metrics
        train_running_correct_predictions = 0.0
        train_loss_epoch = 0.0
        num_traindata = len(data_loader_dict["train"].dataset)

        # Train on batches in current epoch
        for inputs, _, _, _, _, year_labels, _ in tqdm(  # progress bar
            data_loader_dict["train"],
            ncols=100,  # width of progres bar
            desc=f"Training epoch {epoch+1} of {num_epochs}",
        ):
            model.train()

            # Move batch of data onto GPU
            inputs = inputs.to(device=device)
            year_labels = year_labels.to(device=device)

            # Zero the gradient of the loss with respect to the parameters
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  # dim = 1

            # Compute loss on the training data
            loss = loss_fn(outputs, year_labels)

            # Training metrics
            num_batches += 1
            train_loss_batches_list.append(loss.item())
            train_loss_epoch += (loss.item() * inputs.size(0)) / num_traindata
            train_batch_correct_predictions = torch.sum(
                predictions == year_labels.data
            ).item()
            train_running_correct_predictions += train_batch_correct_predictions
            train_accuracy_batches_list.append(
                train_batch_correct_predictions / inputs.size(0)
            )

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

        # Update learning rate (once per epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Training metrics per epoch
        train_accuracy_epochs_list.append(
            train_running_correct_predictions / num_traindata
        )
        train_loss_epochs_list.append(train_loss_epoch)

        # Predict on test set
        inputs, _, _, _, _, year_labels, _ = next(iter(data_loader_dict["test"]))

        with torch.no_grad():  # Track gradients only on training data
            # Move batch of data onto GPU
            inputs = inputs.to(device=device)
            year_labels = year_labels.to(device=device)

            # Predict on test set
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  # dim = 1

            # Compute loss on test set
            loss = loss_fn(outputs, year_labels)

            # Test metrics
            test_loss_epochs_list.append(loss.item())
            test_accuracy_epochs_list.append(
                torch.sum(predictions == year_labels.data).item() / inputs.size(0)
            )

        print(
            f"Train loss: \t{train_loss_epochs_list[-1]:.6f}\t Train accuracy: \t{train_accuracy_epochs_list[-1]:.6f}"
        )
        print(
            f"Test loss:\t{test_loss_epochs_list[-1]:.6f} \t Test accuracy: \t{test_accuracy_epochs_list[-1]:.6f}"
        )
        train_time_epochs_list.append(time.time() - start_time)
        print(
            f"Time up to epoch {epoch+1}: {str(datetime.timedelta(seconds=train_time_epochs_list[-1]))}\n"
        )

    # Store results in dataframes
    results_batches_df = pd.DataFrame(
        {
            "Batch": torch.arange(start=1, end=num_batches + 1).tolist(),
            "Batch Size": batch_size,
            "Loss": train_loss_batches_list,
            "Accuracy": train_accuracy_batches_list,
            "Phase": "Train",
        }
    )

    results_epochs_df = pd.DataFrame(
        {
            "Epoch": torch.arange(start=1, end=num_epochs + 1).tolist(),
            "Loss": train_loss_epochs_list,
            "Accuracy": train_accuracy_epochs_list,
            "Time": train_time_epochs_list,
            "Phase": "Train",
        }
    )
    results_epochs_df = pd.concat(
        [
            results_epochs_df,
            pd.DataFrame(
                {
                    "Epoch": torch.arange(start=1, end=num_epochs + 1).tolist(),
                    "Loss": test_loss_epochs_list,
                    "Accuracy": test_accuracy_epochs_list,
                    "Time": train_time_epochs_list,
                    "Phase": "Test",
                }
            ),
        ]
    )

    return results_batches_df, results_epochs_df


def train(
    year_bucket_size: int = 2,
    train_subset_size: Optional[int] = 10000,
    test_subset_size: Optional[int] = 1000,
    features_path: str = "../raw_data/tables/features.csv",
    batch_size: int = 64,
    num_epochs=10,
    learning_rate: float = 1e-1,
    results_file_path: Optional[str] = "results/",
    model_weights_file_path: Optional[str] = "./",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    seed: int = 100,
):
    # Set seed for random number generator
    # Fixes order in dataloader
    torch.manual_seed(seed)

    # Load all features
    # features_df = data_cleaning.create_feature_df() # recreates feature dataframe
    features_df = pd.read_csv(features_path)

    # Create feature dataframes for train, val, test
    train_df, val_df, test_df = split_dataset_dfs(features_path)

    # Set bucket_size for year -> how many years should correspond to one label?
    min_year = features_df["Launch_Year"].min()  # oldest car launch_year in data
    max_year = features_df["Launch_Year"].max()  # # most recent car launch_year in data

    # Create training, test, and validation torch.dataset
    if train_subset_size is not None:
        # Draw random subset
        perm = torch.randperm(len(train_df))
        idcs = perm[:train_subset_size].numpy()
        train_df = train_df.iloc[idcs]
    if test_subset_size is not None:
        # Draw random subset
        perm = torch.randperm(len(test_df))
        idcs = perm[:test_subset_size].numpy()
        test_df = test_df.iloc[idcs]

    train_set = CarDataset(
        features=train_df,
        transform=data_transforms["train"],
        year2label_fn=lambda year: year2label_fn(
            year,
            min_year=min_year,
            max_year=max_year,
            year_bucket_size=year_bucket_size,
        ),
    )

    test_set = CarDataset(
        features=test_df,
        transform=data_transforms["val"],
        year2label_fn=lambda year: year2label_fn(
            year,
            min_year=min_year,
            max_year=max_year,
            year_bucket_size=year_bucket_size,
        ),
    )

    val_set = CarDataset(
        features=val_df,
        transform=data_transforms["val"],
        year2label_fn=lambda year: year2label_fn(
            year,
            min_year=min_year,
            max_year=max_year,
            year_bucket_size=year_bucket_size,
        ),
    )

    # Data loaders for all train, test, val datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=len(test_set), shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=len(val_set), shuffle=True
    )

    # Dictionary of dataloaders for train, val, test
    data_loader_dict = dict(
        {"train": train_loader, "val": val_loader, "test": test_loader}
    )

    # Instantiate model
    # Compute number of classes of final layer which depends on year_bucket size and year_range
    year_range = max_year - min_year
    num_year_classes = 1 + (year_range // year_bucket_size)  # floor division

    my_model = get_fine_tuneable_model(num_classes=num_year_classes)
    my_model.to(device=device)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device=device)

    # Optimizer
    optimizer = torch.optim.Adam(
        my_model.fc.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # Learning rate
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=0.05,
    )

    # Training loop
    results_batches_df, results_epochs_df = training_loop(
        model=my_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        data_loader_dict=data_loader_dict,
        lr_scheduler=lr_scheduler,
        num_epochs=num_epochs,
        device=device,
    )

    # Save weights of trained model
    if model_weights_file_path is not None:
        torch.save(my_model.state_dict(), model_weights_file_path + "model_weights.pt")

    # Save results to file
    if results_file_path is not None:
        results_batches_df.to_csv(results_file_path + "train_test_results_batches.csv")
        results_epochs_df.to_csv(results_file_path + "train_test_results_epochs.csv")

    return results_batches_df, results_epochs_df


if __name__ == "__main__":
    fire.Fire(train)
