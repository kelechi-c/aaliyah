"""
playground vision project, image classification, dinov2 backbone and tuned linear head.
"""

from jax import numpy as jnp, Array, random as jrand
from flax import nnx
from PIL import Image
import pandas as pd, optax, numpy as np
import jax, optax, wandb, torch, os, click, math, gc, time
from tqdm.auto import tqdm
from functools import partial
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from .dinov2 import DinoViT

jax.config.update("jax_default_matmul_precision", "bfloat16")

from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"


num_devices = jax.device_count()
devices = jax.devices()


print(f"found {num_devices} JAX device(s)")
for device in devices:
    print(f"{device} / ")

mesh_devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(mesh_devices, axis_names=("data",))
data_sharding = NS(mesh, PS("data"))
rep_sharding = NS(mesh, PS())


class ImageData(IterableDataset):
    def __init__(self, images, labels, split=1000):
        self.image = images[:split]
        self.label = labels[:split]
        self.split = split or len(images)

    def __len__(self):
        return self.split

    def __iter__(self):
        for image, label in zip(self.image, self.label):

            image = jnp.array(image, dtype=jnp.bfloat16)
            label = jnp.array(label, dtype=jnp.int32)

            yield {"image": image, "label": label}


def jax_collate(batch):
    latents = jnp.stack([item["image"] for item in batch], axis=0)
    labels = jnp.stack([item["label"] for item in batch], axis=0)

    return {
        "image": latents,
        "label": labels,
    }


rngs = nnx.Rngs(0)
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)


from safetensors.numpy import load_file as loadnp
from flax import nnx
from flax.core.frozen_dict import freeze


def flatten_dict(d, parent_key=(), sep="/"):
    """Flatten a nested dictionary; keys become tuples."""
    items = {}
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def unflatten_dict(flat, sep="/"):
    """Convert a flattened dictionary with tuple keys back to nested form."""
    nested = {}
    for key_tuple, v in flat.items():
        current = nested
        for part in key_tuple[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[key_tuple[-1]] = v
    return nested


def load_model(model, model_file="./model.safetensors"):
    # Load the flat checkpoint dictionary.
    loaded_state = loadnp(model_file)
    loaded_nested = {}
    # First, unflatten the keys (they are strings with the sep character)
    for k, v in loaded_state.items():
        keys = tuple(k.split("/"))
        current = loaded_nested
        for part in keys[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[keys[-1]] = v

    # Obtain the abstract state from the model.
    graphdef, abstract_state = nnx.split(model)
    # Convert the abstract state to a pure dictionary.
    expected_state = nnx.to_pure_dict(abstract_state)

    # Flatten both dictionaries.
    flat_loaded = flatten_dict(loaded_nested)
    flat_expected = flatten_dict(expected_state)

    # Filter the loaded flat dictionary to include only keys that exist in the expected state.
    filtered_flat = {k: v for k, v in flat_loaded.items() if k in flat_expected}

    # Unflatten back to the nested dictionary structure.
    filtered_state = unflatten_dict(filtered_flat)

    # Update the abstract state with the filtered state.
    nnx.replace_by_pure_dict(abstract_state, filtered_state)
    # Merge the updated state with the graph definition.
    model = nnx.merge(graphdef, abstract_state)

    return model, nnx.state(model)


dino_model = DinoViT()
dino_model, state = load_model(dino_model, "/content/model/dinov2-small.safetensors")

linear_adapter = nnx.Linear(384, 10, rngs=rngs)


class Classifier(nnx.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.backbone = dino_model  # DinoViT()
        self.clf_head = linear_adapter
        self.clf_head.train()

    def __call__(self, x):
        x = self.backbone(x)
        x = self.clf_head(x)

        return x


classifier = Classifier()
head_params = nnx.All(nnx.Param, nnx.PathContains("clf_head"))

# nnx.display(classifier)

learn_rate = 1e-4
# classifier.train()
optimizer = nnx.Optimizer(
    classifier, optax.adamw(learning_rate=learn_rate), wrt=head_params
)

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
)


def wandb_logger(key: str, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)
    # wandb.run()


@nnx.jit
def train_step(model, optimizer, metrics: nnx.MultiMetric, batch):
    def loss_func(model, batch):
        image, label = batch
        # print(f"{image.shape = }")
        print(f"{label.shape = }")

        logits = model(image).astype(jnp.float32).reshape((image.shape[0], -1))
        print(f"{logits.shape = } / {logits.dtype}")

        label = label.astype(jnp.int32)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels=label
        ).mean()

        return loss, logits

    diff_state = nnx.DiffState(0, head_params)
    gradfn = nnx.value_and_grad(
        loss_func, has_aux=True, allow_int=True, argnums=diff_state
    )
    (loss, logits), grads = gradfn(model, batch)

    # predictions = jnp.argmax(logits, axis=-1)
    # accuracy = jnp.mean(predictions == batch[1])
    metrics.update(loss=loss, logits=logits, labels=batch[1])

    optimizer.update(grads)
    grad_norm = optax.global_norm(grads)
    acc = metrics.compute()["accuracy"]

    return loss, acc, grad_norm


def trainer(model=classifier, optimizer=optimizer, train_loader=train_loader):
    epochs = 50
    train_loss = 0.0
    accuracy = 0.0

    model.train()

    wandb_logger(
        key="",
        project_name="dino_classifier",
        run_name="test-cifar",
    )

    batch = next(iter(train_loader))

    for epoch in tqdm(range(epochs)):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            train_loss, accuracy, grad_norm = train_step(
                model, optimizer, metrics, batch
            )
            # print(f"step {step}, loss-> {train_loss.item():.4f}, acc {accuracy.item():.4f}")

            wandb.log({"loss": train_loss.item(), "accuracy": accuracy.item()})

            print(
                f"step {step}, train loss {train_loss:.4f}, grad_norm {grad_norm:.4f}, accuracy: {accuracy*100:.4f}"
            )

        print(f"epoch {epoch} complete")


trainer()
