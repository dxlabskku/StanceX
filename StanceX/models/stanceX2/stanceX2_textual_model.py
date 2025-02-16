import inspect
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

class stanceXTextualModel(nn.Module):
    def __init__(self, args, num_layers_to_freeze=0):
        super().__init__()

        self.args = args
        self.config = AutoConfig.from_pretrained(args.textual_transformer_name)
        self.hidden_size = self.config.hidden_size
        self.textual_plm = AutoModel.from_pretrained(args.textual_transformer_name, self.config)
        self.num_layers_to_freeze = num_layers_to_freeze
        self.freeze_layers(self.num_layers_to_freeze)
        
    def freeze_layers(self, num_layers_to_freeze):
        for name, param in self.textual_plm.named_parameters():
            if any(f"layer.{i}." in name for i in range(num_layers_to_freeze)):
                param.requires_grad = False

    def forward(self, input_data):
        outputs = self.textual_plm(
        input_ids=input_data['input_ids'],
        attention_mask=input_data['attention_mask']
        )

        pooled_output = outputs['last_hidden_state'][:, 0, :]
        last_hidden_state = outputs['last_hidden_state']
 
        return {'pooled_output': pooled_output, 'last_hidden_state': last_hidden_state}
       
if __name__ == '__main__':
    class Args():
        textual_transformer_tokenizer_name = 'model_state/roberta-base'
        textual_transformer_name = 'model_state/roberta-base'

    import torch
    args = Args()
    model = stanceXTextualModel(args)
    text_ids = torch.randint(low=0, high=50264, size=[16, 512], dtype=torch.long)
    text_masks = torch.ones(size=[16, 512], dtype=torch.long)
    text_loss_ids = torch.zeros(size=[16, 512], dtype=torch.long)
    text_loss_ids[:, 100] = 1
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'text_loss_ids': text_loss_ids}
    
    logits = model(input_data)
