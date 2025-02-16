import math
import inspect
import torch
import torch.nn as nn
from torch.nn import Dropout
from transformers import AutoConfig, AutoModel

class stanceXVisualModel(nn.Module):
    def __init__(self, args, num_layers_to_freeze=0):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.visual_transformer_name)
        self.visual_plm = AutoModel.from_pretrained(args.visual_transformer_name, self.config)
        self.visual_soft_tokens = args.visual_soft_tokens
        self.hidden_size = self.config.hidden_size

        val = math.sqrt(6. / float(self.hidden_size * 2))
        self.soft_prompt_embeds = nn.Parameter(torch.zeros(1, self.visual_soft_tokens, self.hidden_size))
        nn.init.uniform_(self.soft_prompt_embeds.data, -val, val)
        self.soft_prompt_dropout = nn.Dropout(args.visual_soft_prompt_dropout)


        self.num_layers_to_freeze = num_layers_to_freeze
        self.freeze_layers(self.num_layers_to_freeze)
        
    def freeze_layers(self, num_layers_to_freeze):
        for name, param in self.visual_plm.named_parameters():
            if any(f"layer.{i}." in name for i in range(num_layers_to_freeze)):
                param.requires_grad = False

    def incorporate_prompt(self, pixel_values):
        batch_size = pixel_values.shape[0]
        x = self.visual_plm.embeddings(pixel_values)
        soft_prompt_embeddings = self.soft_prompt_dropout(self.soft_prompt_embeds.expand(batch_size, -1, -1))
        
        x = torch.cat((
                x[:, :1, :],
                soft_prompt_embeddings,
                x[:, 1:, :]
            ), dim=1)
        
        return x

    def forward(self, input_data):
        embedding_output = self.incorporate_prompt(**{k: v for k, v in input_data.items() if k in inspect.signature(self.incorporate_prompt).parameters})
        encoder_outputs = self.visual_plm.encoder(embedding_output)

        last_hidden_state = encoder_outputs['last_hidden_state']
        pooled_output = self.visual_plm.pooler(last_hidden_state)
        soft_hidden_state = self.visual_plm.layernorm(last_hidden_state)[:, 1:1+self.visual_soft_tokens, :]
        soft_hidden_state = torch.avg_pool1d(soft_hidden_state.transpose(1, 2), kernel_size=self.visual_soft_tokens).squeeze(-1)
        
        return {
            'last_hidden_state': soft_hidden_state,
            'pooled_output': pooled_output
        }


if __name__ == '__main__':
    class Args():  
        visual_transformer_name = 'model_state/google/vit-base-patch16-224'
        visual_soft_tokens = 5 
        visual_soft_prompt_dropout = 0.2
        self.num_layers_to_freeze = num_layers_to_freeze
        self.freeze_layers(self.num_layers_to_freeze)

    import torch
    args = Args()

    model = stanceXVisualModel(args)
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'pixel_values': image_tensor}
    logits = model(input_data)
