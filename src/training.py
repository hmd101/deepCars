import fire
import torch
from typing import Optional
import time
import pandas as pd

from data.car_dataset import CarDataset, data_transforms, inverse_transform, year2label_fn

from data.data_splitting import split_dataset_dfs

from data import data_cleaning
from models import get_fine_tuneable_model

def training_loop(model, loss_fn, optimizer, data_loader_dict, num_epochs=3,phase="train", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    acc_per_batch_list = []
    loss_per_batch_list  =[]
    num_batches = 0
    epoch_acc_list = []
    epoch_loss_list = []
    time_list = []

    start_time = time.time()

    for epoch in range(num_epochs):

        # print epoch
        print(f"Epoch {epoch} of {num_epochs -1}")
        print("--" * 10)

        # each epoch has a training and validation phase
        if phase == "train":
            model.train() # set model to training mode

        else:
            model.eval() # set model to evaluation mode

        # for tracking performance
        running_loss = 0.0
        running_corrects = 0.0

        # iterate over data
        ## loop over inputs and labels in dataloaders (in dict form for training/test phase)

        #  a dataloader has  the following attributes: 
        # image, bodytype, model_id, launch_year, self.bodytype2label_fn(bodytype), self.year2label_fn(year=launch_year), viewpoint

        ## inputs and labels to device
        for inputs, _, _, _, _,year_labels, viewpoint in data_loader_dict[phase]:

            #print(inputs.shape)
 
            inputs = inputs.to(device)
            year_labels = year_labels.to(device)
    

            ## zero the parameters of the gradient
            optimizer.zero_grad()

            # forward pass
            # track history only if in train mode
            with torch.set_grad_enabled(phase=="train"):
                outputs = model(inputs)
                _, predictions = torch.max(outputs,1) # dim = 1
                loss = loss_fn(outputs, year_labels)
                # backward + optimize only when in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # Statistics:
            num_corrects = torch.sum(predictions == year_labels.data).item()
            running_corrects += num_corrects
            num_batches += 1
            acc_per_batch = num_corrects / inputs.size(0)
            loss_per_batch = loss.item()
            running_loss  += loss_per_batch
            loss_per_batch_list.append(loss_per_batch)
            acc_per_batch_list.append(acc_per_batch)


        epoch_loss = running_loss / len(data_loader_dict[phase].dataset)
        epoch_acc = running_corrects / len(data_loader_dict[phase].dataset) #  .double() is equivalent to self.to(torch.float64)
        epoch_acc_list.append(epoch_acc)
        epoch_loss_list.append(epoch_loss)
        print(f"{phase}:, Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

        # record training time
        time_list.append(time.time() - start_time)
        print(f"Time up to epoch {epoch}: {time_list[epoch]}")
    
    return list(range(num_batches)), loss_per_batch_list, acc_per_batch_list, list(range(num_epochs)), epoch_loss_list, epoch_acc_list, time_list



def train(year_bucket_size:int = 2, data_subset_size:Optional[int]=10,features_path:str="../raw_data/tables/features.csv",batch_size:int=32, num_epochs=2, learning_rate:float=5e-4,results_file_path:Optional[str]="results/",model_weights_file_path:Optional[str]="./",device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),seed:int=100):
    # set seed for random number generator
    # fixes order in dataloader
    torch.manual_seed(seed)

    # load all features
    #features_df = data_cleaning.create_feature_df() # recreates feature dataframe
    features_df = pd.read_csv(features_path)

    # create feature dataframes for train, val, test
    train_df, val_df, test_df = split_dataset_dfs(features_path)

    # set bucket_size for year -> how many years should correspond to one label?
    min_year = features_df["Launch_Year"].min() # oldest car launch_year in data
    max_year = features_df["Launch_Year"].max() # # most recent car launch_year in data

    # create training, test, and validation torch.dataset 
    if data_subset_size is not None:
        train_df = train_df[:data_subset_size]
        val_df = val_df[:data_subset_size]
        test_df = test_df[:data_subset_size]
        
    train_set = CarDataset(features=train_df, transform=data_transforms["train"],year2label_fn=lambda year:year2label_fn(year, min_year = min_year, max_year = max_year, year_bucket_size=year_bucket_size))

    test_set = CarDataset(features=test_df, transform=data_transforms["val"],year2label_fn=lambda year:year2label_fn(year, min_year = min_year, max_year = max_year, year_bucket_size=year_bucket_size))

    val_set = CarDataset(features=val_df, transform=data_transforms["val"],year2label_fn=lambda year:year2label_fn(year, min_year = min_year, max_year = max_year, year_bucket_size=year_bucket_size))



    # data loaders for all train, test, val datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)

    # dictionary of dataloaders for train, val, test
    data_loader_dict = dict({"train":train_loader, "val":val_loader, "test":test_loader})

    # instantiate model
    # compute number of classes of final layer which depends on year_bucket size and year_range
    year_range = max_year - min_year
    num_year_classes = 1 +  (year_range // year_bucket_size) # floor division

    my_model = get_fine_tuneable_model(num_classes=num_year_classes)
    my_model.to(device=device)
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device=device)

    # optimizer
    optimizer = torch.optim.SGD(my_model.fc.parameters(), lr=learning_rate)

    # training loop
    train_num_batches_lst, train_loss_per_batch_lst, train_acc_per_batch_lst, train_num_epochs_lst, train_epoch_loss_lst, train_epoch_acc_lst, train_time_list = training_loop(model=my_model, loss_fn=loss_fn, optimizer=optimizer, data_loader_dict=data_loader_dict, num_epochs = num_epochs,device=device, phase="train")

    # evaluate model on test set
    test_num_batches_lst, test_loss_per_batch_lst, test_acc_per_batch_lst, test_num_epochs_lst, test_epoch_loss_lst, test_epoch_acc_lst, test_time_list = training_loop(model=my_model, loss_fn=loss_fn, optimizer=optimizer, data_loader_dict=data_loader_dict, num_epochs = num_epochs,device=device, phase="test")

    # results as csv
    df_train_results_batches = pd.DataFrame({"batch":train_num_batches_lst, "loss":train_loss_per_batch_lst, "accuracy":train_acc_per_batch_lst})

    df_train_results_epochs = pd.DataFrame({"epoch":train_num_epochs_lst,"time": train_time_list,"loss":train_epoch_loss_lst, "accuracy":train_epoch_acc_lst})

    df_test_results_batches = pd.DataFrame({"batch":test_num_batches_lst, "loss":test_loss_per_batch_lst, "accuracy":test_acc_per_batch_lst})

    df_test_results_epochs = pd.DataFrame({"epoch":test_num_epochs_lst,"time": test_time_list,"loss":test_epoch_loss_lst, "accuracy":test_epoch_acc_lst})   


    # save weights of trained model
    if model_weights_file_path is not None:
        torch.save(my_model.state_dict(), model_weights_file_path + "model_weights.pt")

    # save results to file
    if results_file_path is not None:
        df_train_results_batches.to_csv(results_file_path+"train_results_batches.csv")
        df_train_results_epochs.to_csv(results_file_path+"train_results_epochs.csv")
        df_test_results_batches.to_csv(results_file_path+"test_results_batches.csv")
        df_test_results_epochs.to_csv(results_file_path+"test_results_epochs.csv")
    
    return df_train_results_batches, df_train_results_epochs, df_test_results_batches, df_test_results_epochs
   

if __name__ == "__main__":
    fire.Fire(train)