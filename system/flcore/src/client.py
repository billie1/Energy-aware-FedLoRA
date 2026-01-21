import torch
from transformers import AutoModelForImageClassification
from lora import LoRALayer  # 假设LoRALayer已定义

class Client:
    def __init__(self, client_id, device, data_loader, lora_rank, lora_alpha, lora_dropout):
        self.client_id = client_id
        self.device = device
        self.data_loader = data_loader
        self.model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model.to(device)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.add_lora_layers()

    def add_lora_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                input_dim = module.weight.shape[1]
                output_dim = module.weight.shape[0]
                if input_dim == 768 and output_dim == 768:
                    lora_layer = LoRALayer(input_dim, output_dim, self.lora_rank, self.lora_alpha, self.lora_dropout)
                    setattr(self.model, f'lora_{name}', lora_layer)

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(5):  # 训练5个epoch
            epoch_loss = 0.0
            for batch in self.data_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids)
                loss = criterion(outputs.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Client {self.client_id} Epoch {epoch}: Loss = {epoch_loss}")
        
        return self.model.state_dict()  # 返回本地训练的模型参数
