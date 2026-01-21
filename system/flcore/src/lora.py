class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, r, lora_alpha, lora_dropout=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_A = nn.Parameter(torch.randn(input_dim, self.r))
        self.lora_B = nn.Parameter(torch.zeros((self.r, output_dim)))
        self.scaling = self.lora_alpha / self.r

    def forward(self, x):
        original_shape = x.shape
        x_flattened = x.view(-1, original_shape[-1])
        device = x_flattened.device

        # Move lora_A and lora_B to the same device as x_flattened
        self.lora_A.data = self.lora_A.data.to(device)
        self.lora_B.data = self.lora_B.data.to(device)
        lora_output = self.lora_dropout(
            x_flattened @ self.lora_A) @ self.lora_B * self.scaling
        lora_output = lora_output.view(*original_shape[:-1], -1)

        if lora_output.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"Dimension mismatch: lora_output.shape={lora_output.shape}, x.shape={x.shape}")

        output = x + lora_output
        return output