import torch


class config:
    lr = 1e-4
    epoch_count = 50
    image_size = 224
    grad_acc_step = 4
    batch_size = 32
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safetensor_file = "har_mobilenet.safetensors"
    leaf_dataset_id = "yusuf802/plant-images"
    har_data_id = "Bingsu/Human_Action_Recognition"
    har_model_id = "tensorkelechi/HAR_mobilenet"
    plant_model_id = "tensorkelechi/leaf_disease_mobilenet"


# Model parameter count
def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count
