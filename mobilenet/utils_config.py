import torch


class config:
    lr = 1e-4
    image_size = 224
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leaf_dataset_id = "yusuf802/plant-images"
    har_data_id = "Bingsu/Human_Action_Recognition"
    dtype = torch.float16
    har_model_id = "tensorkelechi/HAR_mobilenet"
    plant_model_id = "tensorkelechi/leaf_disease_mobilenet"
