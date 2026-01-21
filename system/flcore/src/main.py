import os
import torch
import torch.backends.cuda
import torch.backends.cudnn
from pytorch_lightning.cli import LightningCLI
from src.data import DataModule
from src.model import ClassificationModel
from lora import LoRALayer



class FederatedLearning:
    def __init__(self, server, rounds=10):
        self.server = server
        self.rounds = rounds

    def train(self):
        for round in range(self.rounds):
            print(f"Federated Learning Round {round}")
            updates = []

            # 让所有客户端进行本地训练
            for client in self.server.clients:
                print(f"Training client {client.client_id}...")
                local_model_params = client.train()
                updates.append(local_model_params)

            # 服务器端聚合
            self.server.aggregate_parameters(updates)

            # 服务器端更新模型后，重新分发给客户端
            self.server.distribute_model()


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    CSVLogger, save_dir="output", name="default"
                ),
                "model_checkpoint.monitor": "val_acc",
                "model_checkpoint.mode": "max",
                "model_checkpoint.filename": "best-step-{step}-{val_acc:.4f}",
                "model_checkpoint.save_last": True,
            }
        )
        parser.link_arguments("data.size", "model.image_size")
        parser.link_arguments(
            "data.num_classes", "model.n_classes", apply_on="instantiate"
        )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

cli = MyLightningCLI(
    ClassificationModel,
    DataModule,
    save_config_kwargs={"overwrite": True},
    trainer_defaults={"check_val_every_n_epoch": None},
)

# Federated Learning setup
clients = [Client(client_id=i, device='cuda', data_loader=train_loader, lora_rank=8, lora_alpha=1.0, lora_dropout=0.1) for i in range(3)]
global_model = ClassificationModel()

server = Server(global_model, clients)
federated_learning = FederatedLearning(server, rounds=10)
federated_learning.train()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Federated Learning time: {elapsed_time} seconds")





