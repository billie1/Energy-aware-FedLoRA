class Server:
    def __init__(self, global_model, clients):
        self.global_model = global_model
        self.clients = clients

    def aggregate_parameters(self, updates):
        # 假设updates是一个包含每个客户端模型参数的列表
        global_state_dict = self.global_model.state_dict()

        for key in global_state_dict:
            if key.startswith('lora_'):  # 仅更新lora层的参数
                aggregated_param = torch.mean(torch.stack([update[key] for update in updates]), dim=0)
                global_state_dict[key] = aggregated_param
        
        self.global_model.load_state_dict(global_state_dict)
        print("Global model parameters updated.")

    def distribute_model(self):
        # 将全局模型发送到所有客户端
        for client in self.clients:
            client.model.load_state_dict(self.global_model.state_dict())
