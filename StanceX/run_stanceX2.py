import sys
sys.path.append('./')
import copy
import os
import numpy as np
import logging
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils.tools import set_seed, read_yaml, AttributeDict, save_results
transformers.logging.set_verbosity_error()
from torch.nn import functional as F

from skimage.measure import label, regionprops

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug_mode', action='store_true')
parser.add_argument('--cuda_idx', type=int)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--framework_name', type=str, default='stanceX')
parser.add_argument('--model_name', type=str, default='bert_vit')
parser.add_argument('--fine_tune', action='store_true')
parser.add_argument('--in_target', action='store_true')
parser.add_argument('--zero_shot', action='store_true')
parser.add_argument('--sweep', action='store_true')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--normal', action='store_true')
GLOBAL_ARGS = parser.parse_args()

# Set your wandb project
RUN_FRAMEWORK_NAME = ''
PROJECT_NAME = ''

if GLOBAL_ARGS.zero_shot:
    DATASET_NAME = GLOBAL_ARGS.dataset_name + '_zero_shot'
elif GLOBAL_ARGS.in_target:
    DATASET_NAME = GLOBAL_ARGS.dataset_name + '_in_target'

from utils.dataset_utils.data_config import data_configs as data_configs
from utils.dataset_utils.data_config import datasets as datasets
data_config = data_configs[GLOBAL_ARGS.dataset_name]

SEED = [42, 67, 2022, 31, 15]

SAVE_STATE = False
SAVE_TEST_RESULT = True
if GLOBAL_ARGS.debug_mode:
    print('=====================debug_mode=======================')
    SAVE_STATE = False
    SAVE_TEST_RESULT = False
elif GLOBAL_ARGS.normal:
    pass
else:
    import wandb
    wandb.login()

FILE_PATH = __file__
OUTPUT_DIR = os.path.dirname(__file__) + '/outputs'
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
CUR_DIR = OUTPUT_DIR + f'/{DATASET_NAME}_{GLOBAL_ARGS.framework_name}_{GLOBAL_ARGS.model_name}_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

CUR_MODELS_DIR = CUR_DIR + '/models'
CUR_RESULTS_DIR = CUR_DIR + '/results'
if not GLOBAL_ARGS.debug_mode:
    if not os.path.isdir(CUR_DIR):
        os.makedirs(CUR_DIR)
        os.makedirs(CUR_MODELS_DIR)
        os.makedirs(CUR_RESULTS_DIR)

CONFIG_PATH = {
    'stanceX': 'config/stanceX_config.yaml',
    'stanceX_gpt_cot': 'config/stanceX_config.yaml',
}
raw_config = read_yaml(CONFIG_PATH[GLOBAL_ARGS.framework_name])

if GLOBAL_ARGS.sweep:
    SAVE_STATE = False
    SAVE_TEST_RESULT = False

if GLOBAL_ARGS.model_name in raw_config['model_config']:
    raw_config['model_config'] = raw_config['model_config'][GLOBAL_ARGS.model_name]
else:
    raise ValueError('Invalid model')
raw_config['model_config']['label_size']['value'] = len(data_config.label2idx)

if GLOBAL_ARGS.fine_tune:
    raw_config['train_config'] = raw_config['train_config']['fine_tune_config']
elif GLOBAL_ARGS.in_target:
    raw_config['train_config'] = raw_config['train_config']['in_target_train_config']
elif GLOBAL_ARGS.zero_shot:
    raw_config['train_config'] = raw_config['train_config']['zero_shot_train_config']

if GLOBAL_ARGS.framework_name == 'stanceX' or GLOBAL_ARGS.framework_name == 'stanceX_gpt_cot':
    from models import stanceXModel as Model
else:
    raise ValueError('Invalid framework')

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(filename)-20s : %(lineno)s line - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)
if not GLOBAL_ARGS.debug_mode:
    fhlr = logging.FileHandler(CUR_DIR + '/training_result.log', mode='w')
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)

def calculate_embedding_quality(tweet_outputs, image_outputs, labels):
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    intra_class_distances = []
    inter_class_distances = []
    labels = labels.cpu().numpy()
    for class_label in set(labels):
        class_indices = np.where(labels == class_label)[0]
        other_indices = np.where(labels != class_label)[0]
        
        for i in range(len(class_indices)):
            for j in range(i + 1, len(class_indices)):
                intra_class_distances.append(
                    cosine_similarity(tweet_outputs[class_indices[i]], tweet_outputs[class_indices[j]]).item()
                )
        
        for i in class_indices:
            for j in other_indices:
                inter_class_distances.append(
                    cosine_similarity(tweet_outputs[i], tweet_outputs[j]).item()
                )

    intra_avg = np.mean(intra_class_distances)
    inter_avg = np.mean(inter_class_distances)
    logging.info(f"Intra-class similarity: {intra_avg:.4f}")
    logging.info(f"Inter-class similarity: {inter_avg:.4f}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        if inputs.ndim == 3:
            inputs = inputs.mean(dim=1)
        if targets.ndim == 2:
            targets = torch.argmax(targets, dim=-1)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def contrastive_loss(tweet_outputs, image_outputs, labels, temperature=0.07):
    tweet_embeds = F.normalize(tweet_outputs, dim=-1)
    image_embeds = F.normalize(image_outputs, dim=-1)

    tweet_similarity = torch.matmul(tweet_embeds, tweet_embeds.T) / temperature
    image_similarity = torch.matmul(image_embeds, image_embeds.T) / temperature

    inter_modal_similarity = torch.matmul(tweet_embeds, image_embeds.T) / temperature

    labels = labels.to(tweet_outputs.device)
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

    pos_mask = label_matrix.float()

    neg_mask = 1 - pos_mask

    # Intra-modal loss
    intra_loss_tweet = F.cross_entropy(tweet_similarity, labels)
    intra_loss_image = F.cross_entropy(image_similarity, labels)

    # Inter-modal loss
    inter_loss = F.cross_entropy(inter_modal_similarity, torch.arange(tweet_outputs.size(0), device=tweet_outputs.device))

    # Combine losses
    total_loss = (intra_loss_tweet + intra_loss_image) * 0.5 + inter_loss
    return total_loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim).to(device))

    def forward(self, embeddings, labels):
        centers_batch = self.centers[labels]
        loss = torch.mean((embeddings - centers_batch) ** 2)
        return loss


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

LABELS = data_config.test_stance
DEVICE = torch.device('cuda:' + str(GLOBAL_ARGS.cuda_idx))


def build_dataset(args, tokenizer, target_name=None):
    train_dataset_orig = datasets[GLOBAL_ARGS.framework_name](
        data_config,
        tokenizer=tokenizer,
        target_name=target_name,
        if_split_hash_tag=args.if_split_hash_tag,
        if_cat_cot=True if GLOBAL_ARGS.framework_name == 'stanceX_gpt_cot' else False,
        in_target=GLOBAL_ARGS.in_target,
        zero_shot=GLOBAL_ARGS.zero_shot,
        train_data=True,
        debug_mode=GLOBAL_ARGS.debug_mode)
    valid_dataset_orig = datasets[GLOBAL_ARGS.framework_name](
        data_config,
        tokenizer=tokenizer,
        target_name=target_name,
        if_split_hash_tag=args.if_split_hash_tag,
        if_cat_cot=True if GLOBAL_ARGS.framework_name == 'stanceX_gpt_cot' else False,
        in_target=GLOBAL_ARGS.in_target,
        zero_shot=GLOBAL_ARGS.zero_shot,
        valid_data=True,
        debug_mode=GLOBAL_ARGS.debug_mode)
    test_dataset_orig = datasets[GLOBAL_ARGS.framework_name](
        data_config,
        tokenizer=tokenizer,
        target_name=target_name,
        if_split_hash_tag=args.if_split_hash_tag,
        if_cat_cot=True if GLOBAL_ARGS.framework_name == 'stanceX_gpt_cot' else False,
        in_target=GLOBAL_ARGS.in_target,
        zero_shot=GLOBAL_ARGS.zero_shot,
        test_data=True,
        debug_mode=GLOBAL_ARGS.debug_mode)

    train_data_loader = DataLoader(
        train_dataset_orig,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    valid_data_loader = DataLoader(
        valid_dataset_orig,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4)
    test_data_loader = DataLoader(
        test_dataset_orig,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4)
    return train_data_loader, valid_data_loader, test_data_loader


def train(train_data_loader, model, contrastive_criterion, center_criterion, focal_criterion, optimizer, scheduler, epoch, num_epochs):
    model.train()
    loss_mean = 0.0

    epoch_contrastive_loss = 0.0
    epoch_center_loss = 0.0
    epoch_focal_loss = 0.0

    # ABLATION
    use_dynamic_weight = True
    use_contrastive = True
    use_center = True
    use_focal = True 

    alpha = max(1 - epoch / (0.5 * num_epochs), 0)
    beta = 1 - alpha

    for train_data in train_data_loader:
        train_data = {k: v.to(DEVICE) for k, v in train_data.items()}
        y = train_data['classification_label']

        tweet_outputs = model.stanceX_textual_model(train_data)['pooled_output']
        image_outputs = model.stanceX_visual_model(train_data)['pooled_output']
        optimizer.zero_grad()

        logits = model(train_data)

        if use_contrastive:
            contrastive_loss_value = contrastive_criterion(tweet_outputs, image_outputs, y)
            epoch_contrastive_loss += contrastive_loss_value.item()
        else:
            contrastive_loss_value = 0.0

        if use_center:
            center_loss_tweet = center_criterion(tweet_outputs, y)
            center_loss_image = center_criterion(image_outputs, y)
            center_loss_value = (center_loss_tweet + center_loss_image) / 2
            epoch_center_loss += center_loss_value.item()
        else:
            center_loss_value = 0.0

        if use_focal:
            focal_loss_value = focal_criterion(logits, y)
            epoch_focal_loss += focal_loss_value.item()
        else:
            focal_loss_value = 0.0

        if use_dynamic_weight:
            total_loss = alpha * ( contrastive_loss_value + center_loss_value) + beta * focal_loss_value
        else:
            total_loss = contrastive_loss_value + center_loss_value +  focal_loss_value

        loss_mean += total_loss.item()

        total_loss.backward()
        optimizer.step()
        scheduler.step()

    logging.info(f"Epoch {epoch}/{num_epochs} - Contrastive Loss: {epoch_contrastive_loss / len(train_data_loader):.4f}, "
                 f"Center Loss: {epoch_center_loss / len(train_data_loader):.4f}, "
                 f"Focal Loss: {epoch_focal_loss / len(train_data_loader):.4f}, "
                 f"Total Loss: {loss_mean / len(train_data_loader):.4f}")
    return loss_mean / len(train_data_loader)


@torch.no_grad()
def evaluate(eval_data_loader, model, criterion, tokenizer):
    model.eval()

    loss_mean = 0.0

    true_labels = []
    predict_labels = []

    textual_tokenizer, visual_processor = tokenizer

    for eval_data in eval_data_loader:
        eval_data = {k: v.to(DEVICE) for k, v in eval_data.items()}
        y = eval_data['classification_label']

        image = eval_data['pixel_values'][0].cpu().permute(1, 2, 0).numpy()

        logits = model(eval_data)
        if logits.ndim == 3:
            logits = logits[:, 0, :]

        loss = criterion(logits, y)
        loss_mean += loss.item()

        preds = logits.argmax(dim=1)

        true_labels.extend(y.cpu().tolist())
        predict_labels.extend(preds.cpu().tolist())

    true_labels = torch.tensor(true_labels).cpu()
    predict_labels = torch.tensor(predict_labels).cpu()

    f1 = f1_score(true_labels, predict_labels, average='macro', labels=LABELS) * 100
    precision = precision_score(true_labels, predict_labels, average='macro', labels=LABELS) * 100
    recall = recall_score(true_labels, predict_labels, average='macro', labels=LABELS) * 100
    acc = accuracy_score(true_labels, predict_labels) * 100



    return (true_labels.tolist(), predict_labels.tolist()), loss_mean / len(eval_data_loader), acc, f1, precision, recall

def train_process(args, train_data_loader, valid_data_loader, test_data_loader, fold=0, target_name=None, tokenizer=None):
    logging.info('init models, optimizer, criterion...')
    model = Model(args).to(DEVICE)

    contrastive_criterion = contrastive_loss
    center_criterion = CenterLoss(num_classes=args.label_size, embedding_dim=model.textual_hidden_size, device=DEVICE).to(DEVICE)
    focal_criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma).to(DEVICE) #, weight=class_weights).to(DEVICE)

    transformer_identifiers = ['shared.weight', 'embedding', 'encoder', 'decoder', 'pooler']
    linear_identifiers = ['linear', 'classifier']
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

    grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in transformer_identifiers) and
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': args.transformer_learning_rate,
         'weight_decay': args.weight_decay},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in transformer_identifiers) and
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': args.transformer_learning_rate,
         'weight_decay': 0.0},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in linear_identifiers) and 
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': args.linear_learning_rate,
         'weight_decay': args.weight_decay},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in linear_identifiers) and 
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': args.linear_learning_rate,
         'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(grouped_model_parameters, lr=args.linear_learning_rate)
    total_steps = len(train_data_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=total_steps * args.warmup_ratio,
                                                num_training_steps=total_steps)

    logging.info('Start training...')
    best_state = None
    best_valid_f1 = 0.0
    best_test_f1 = 0.0
    best_test_acc = 0.0
    best_test_precision = 0.0
    best_test_recall = 0.0
    temp_path = CUR_MODELS_DIR + f'/{DATASET_NAME}_temp_model.pt'
    temp_results_path = CUR_RESULTS_DIR + f'/{DATASET_NAME}_{target_name}_temp_test_results.csv'


    for ep in range(args.num_epochs):
        logging.info(f'epoch {ep+1} start train')

        # Dynamic weighting for Hybrid Loss !!
        alpha = max(1 - ep / (0.5 * args.num_epochs), 0)  # Contrastive & Center Loss 비중
        beta = 1 - alpha 

        train_loss = train(
            train_data_loader, 
            model, 
            contrastive_criterion, 
            center_criterion, 
            focal_criterion, 
            optimizer, 
            scheduler, 
            epoch=ep + 1, 
            num_epochs=args.num_epochs
        )

        logging.info(f'epoch {ep+1} start evaluate')
        evaluation_criterion = nn.CrossEntropyLoss().to(DEVICE)#weight=class_weights).to(DEVICE)

        valid_results, valid_loss, valid_acc, valid_f1, valid_precision, valid_recall = evaluate(valid_data_loader, model, evaluation_criterion, tokenizer=tokenizer)
        test_results, test_loss, test_acc, test_f1, test_precision, test_recall = evaluate(test_data_loader, model, evaluation_criterion, tokenizer=tokenizer)
        
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_test_f1 = test_f1
            best_test_acc = test_acc
            best_test_precision = test_precision
            best_test_recall = test_recall
            best_path = CUR_MODELS_DIR + \
                f'/{DATASET_NAME}_{target_name}_fold_{fold}_{best_test_f1:.5f}_{datetime.now().strftime("%m-%d-%H-%M")}.pt'
            best_results_path = CUR_RESULTS_DIR + \
                f'/{DATASET_NAME}_{target_name}_fold_{fold}_{best_test_f1:.5f}_{datetime.now().strftime("%m-%d-%H-%M")}.csv'
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_results = copy.deepcopy(test_results)

            # Save the best model and results
            if not GLOBAL_ARGS.debug_mode and not GLOBAL_ARGS.normal:
                wandb.log({
                    f'{fold}_epoch': ep + 1, 
                    "best_valid_f1": best_valid_f1, 
                    "best_test_f1": best_test_f1,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_loss": test_loss,
                    "contrastive_weight": alpha,
                    "focal_weight": beta
                })
            if ep > 3:
                logging.info(f'Best model acc {best_test_acc:.5f}, f1 {best_test_f1:.5f}, precision {best_test_precision:.5f}, recall {best_test_recall:.5f}')
                if best_state is not None and SAVE_STATE:
                    logging.info(f'Saving best model in {temp_path}')
                    torch.save(best_state, temp_path)
                if SAVE_TEST_RESULT:
                    logging.info(f'Saving best results in {temp_path}')
                    save_results(best_results, temp_results_path)

        if not GLOBAL_ARGS.debug_mode and not GLOBAL_ARGS.normal:
            wandb.log({f'{fold}_epoch': ep+1, "train_loss": train_loss, "valid_loss": valid_loss, "test_loss": test_loss, "valid_f1": valid_f1, "test_f1": test_f1})
        logging.info(f'epoch {ep+1} done! train_loss {train_loss:.5f}, valid_loss {valid_loss:.5f}, test_loss {test_loss:.5f}')
        logging.info(f'valid: acc {valid_acc:.5f}, f1 {valid_f1:.5f}, precision {valid_precision:.5f}, recall {valid_recall:.5f}')
        logging.info(f'test: acc {test_acc:.5f}, f1 {test_f1:.5f}, precision {test_precision:.5f}, recall {test_recall:.5f}, now best_f1 {best_test_f1:.5f}')

        if (ep + 1) % 1 == 0:  # 매 에폭마다 실행
            all_tweet_outputs = []
            all_image_outputs = []
            all_labels = []

            for eval_data in test_data_loader:
                eval_data = {k: v.to(DEVICE) for k, v in eval_data.items()}
                y = eval_data['classification_label']
                tweet_outputs = model.stanceX_textual_model(eval_data)['pooled_output']
                image_outputs = model.stanceX_visual_model(eval_data)['pooled_output']

                all_tweet_outputs.append(tweet_outputs.cpu().detach())
                all_image_outputs.append(image_outputs.cpu().detach())
                all_labels.append(y.cpu().detach())

            all_tweet_outputs = torch.cat(all_tweet_outputs, dim=0)
            all_image_outputs = torch.cat(all_image_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            logging.info(f"\nEpoch {ep + 1} Embedding Quality Metrics:")
            logging.info("Text Embeddings Quality:")
            calculate_embedding_quality(all_tweet_outputs, all_image_outputs, all_labels)
            
            logging.info("\nImage Embeddings Quality:")
            calculate_embedding_quality(all_image_outputs, all_tweet_outputs, all_labels)

    logging.info(f'Best model acc {best_test_acc:.5f}, f1 {best_test_f1:.5f}, precision {best_test_precision:.5f}, recall {best_test_recall:.5f}')
    if best_state != None and SAVE_STATE:
        torch.save(best_state, best_path)
    if best_state != None and SAVE_STATE:
        logging.info(f'Saving best model in {best_path}')
        torch.save(best_state, best_path)
    if SAVE_TEST_RESULT:
        logging.info(f'Saving best results in {best_results_path}')
        save_results(best_results, best_results_path)

    return best_results, best_test_acc, best_test_f1, best_test_precision, best_test_recall

def main(args=None):
    if not GLOBAL_ARGS.debug_mode and not GLOBAL_ARGS.normal:
        run_name = f'{RUN_FRAMEWORK_NAME}_{DATASET_NAME}_{GLOBAL_ARGS.framework_name}_{GLOBAL_ARGS.model_name}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
        wandb.init(project=PROJECT_NAME, name=run_name, config=args)
        args = wandb.config
    set_seed()
    logging.info('Using cuda device gpu: ' + str(GLOBAL_ARGS.cuda_idx))
    if not GLOBAL_ARGS.debug_mode:
        logging.info('Saving into directory ' + CUR_DIR)

    output_str = ''
    final_score = 0
    targeted_result = {}
    target_names = data_config.target_names if not GLOBAL_ARGS.zero_shot else data_config.zero_shot_target_names
    for target_name in target_names:
        results = []
        acc = []
        f1 = []
        precision = []
        recall = []
        logging.info('preparing data...')
        textual_tokenizer = AutoTokenizer.from_pretrained(args.textual_transformer_tokenizer_name, model_max_length=args.max_tokenization_length, use_fast=False)
        visual_processor = AutoImageProcessor.from_pretrained(args.visual_transformer_tokenizer_name)
        tokenizer = (textual_tokenizer, visual_processor)
        for fold in range(args.train_times):
            logging.info(f'all train times:{args.train_times}, now is fold:{fold+1}')
            logging.info('preparing data...')
            set_seed(SEED[fold])

            train_data_loader, valid_data_loader, test_data_loader = build_dataset(args, tokenizer, target_name)
            best_results, best_acc, best_f1, best_precision, best_recall = train_process(args, train_data_loader, valid_data_loader, test_data_loader, fold=fold, target_name=data_config.short_target_names[target_name], tokenizer=tokenizer)
            results.append(best_results)
            acc.append(best_acc)
            f1.append(best_f1)
            precision.append(best_precision)
            recall.append(best_recall)

            logging.info(f'target "{data_config.short_target_names[target_name]}" fold {fold+1} train finish, acc:{best_acc:.5f}, f1:{best_f1:.5f}, precision:{best_precision:.5f}, recall:{best_recall:.5f}')
        targeted_result[data_config.short_target_names[target_name]] = results
        acc_avg = np.mean(acc)
        f1_avg = np.mean(f1)
        precision_avg = np.mean(precision)
        recall_avg = np.mean(recall)
        acc_std = np.std(acc, ddof=0)
        f1_std = np.std(f1, ddof=0)
        precision_std = np.std(precision, ddof=0)
        recall_std = np.std(recall, ddof=0)
        acc_sem = acc_std / np.sqrt(args.train_times)
        f1_sem = f1_std / np.sqrt(args.train_times)
        precision_sem = precision_std / np.sqrt(args.train_times)
        recall_sem = recall_std / np.sqrt(args.train_times)

        logging.info(f'target "{data_config.short_target_names[target_name]}" train finish')
        result = {'target_name': data_config.short_target_names[target_name], 'acc_avg': acc_avg, 'f1_avg': f1_avg, 'precision_avg': precision_avg, 'recall_avg': recall_avg, 'acc_std': acc_std, 'f1_std': f1_std, 'precision_std': precision_std, 'recall_std': recall_std, 'acc_sem': acc_sem, 'f1_sem': f1_sem, 'precision_sem': precision_sem, 'recall_sem': recall_sem}
        final_score += f1_avg
        for key in result.values():
            if type(key) is str:
                output_str += f'{key}\t'
            else:
                output_str += f'{key:.5f}\t'
        output_str += '\n'

    # calc over all target results
    acc = []
    f1 = []
    precision = []
    recall = []
    for fold in range(args.train_times):
        overall_true = []
        overall_pred = []
        for target_name in target_names:
            true_labels, predict_labels = targeted_result[data_config.short_target_names[target_name]][fold]
            overall_true += true_labels
            overall_pred += predict_labels
        acc.append(accuracy_score(overall_true, overall_pred) * 100)
        f1.append(f1_score(overall_true, overall_pred, average='macro', labels=LABELS) * 100)
        precision.append(precision_score(overall_true, overall_pred, average='macro', labels=LABELS) * 100)
        recall.append(recall_score(overall_true, overall_pred, average='macro', labels=LABELS) * 100)
    acc_avg = np.mean(acc)
    f1_avg = np.mean(f1)
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    acc_std = np.std(acc, ddof=0)
    f1_std = np.std(f1, ddof=0)
    precision_std = np.std(precision, ddof=0)
    recall_std = np.std(recall, ddof=0)
    acc_sem = acc_std / np.sqrt(args.train_times)
    f1_sem = f1_std / np.sqrt(args.train_times)
    precision_sem = precision_std / np.sqrt(args.train_times)
    recall_sem = recall_std / np.sqrt(args.train_times)

    result = {'target_name': 'overall', 'acc_avg': acc_avg, 'f1_avg': f1_avg, 'precision_avg': precision_avg, 'recall_avg': recall_avg, 'acc_std': acc_std, 'f1_std': f1_std, 'precision_std': precision_std, 'recall_std': recall_std, 'acc_sem': acc_sem, 'f1_sem': f1_sem, 'precision_sem': precision_sem, 'recall_sem': recall_sem}
    overall_f1 = f1_avg
    for key in result.values():
        if type(key) is str:
            output_str += f'{key}\t'
        else:
            output_str += f'{key:.5f}\t'
    output_str += '\n'

    logging.info(f'all train finish')
    output_str = f'\ntarget_name\tacc_avg:\tf1_avg:\tprecision_avg:\trecall_avg:\tacc_std\tf1_std\tprecision_std\trecall_std\tacc_sem\tf1_sem\tprecision_sem\trecall_sem\n' + output_str
    logging.info(output_str)
    final_score = final_score / len(target_names)
    if not GLOBAL_ARGS.debug_mode and not GLOBAL_ARGS.normal:
        wandb.log({'final_score': final_score, 'overall_f1': overall_f1})


if __name__ == '__main__':
    logging.info(f'pid: {os.getpid()}')
    
    if GLOBAL_ARGS.debug_mode or GLOBAL_ARGS.normal:
        if GLOBAL_ARGS.debug_mode:
            raw_config['train_config']['train_times']['value'] = 1
            raw_config['train_config']['num_epochs']['value'] = 2
        config = {}
        for value in raw_config.values():
            for k, v in value.items():
                if 'value' in v:
                    config[k] = v['value']
                elif 'values' in v:
                    config[k] = v['values'][0]
        config = AttributeDict(config)
        main(config)
    elif GLOBAL_ARGS.sweep:
        sweep_config = {
            'method': 'random',
            'metric': {'goal': 'maximize', 'name': 'final_score'},
            'parameters': {}
        }
        for v in raw_config.values():
            sweep_config['parameters'].update(v)
        sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
        wandb.agent(sweep_id, main, count=50)
    else:
        config = {}
        for value in raw_config.values():
            for k, v in value.items():
                if 'value' in v:
                    config[k] = v['value']
                elif 'values' in v:
                    config[k] = v['values'][0]
        main(config)