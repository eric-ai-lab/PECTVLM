import sys;sys.path.append("/data1/zhenzhang/dir2/naive_finetuning");sys.path.append("/data1/zhenzhang/dir2/lib")
from Multilingual_CLIP.multilingual_clip import pt_multilingual_clip
from trainer import train, estimate
from load_datasets import load_file_from_task,load_file,load_file_task2
import clip
import torch
import transformers
from transformers import set_seed
import os
import argparse
import time
import json
from mylogging import add_filehandler, get_logger
from util import get_model_state_dict_requires_grad
os.environ['TOKENIZERS_PARALLELISM'] = "false"

logger = get_logger(__name__)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str,choices=['en','de','fr','cs'],default='en')
    
    parser.add_argument('--language_list', nargs='+')
    # parser.add_argument('--language2', type=str,choices=['en','de','fr','cs'],default='de')
    # parser.add_argument('--tolanguage', type=str,choices=['en','de'])
    # parser.add_argument('--joint_lang_train', action='store_true')
    parser.add_argument('--delta_tuning', action='store_true')
    parser.add_argument('--if_weight', action='store_true')
    # parser.add_argument('--delta_type', type=str,choices=['prompt','lora','ft','c_lora'],default='ft')
    parser.add_argument('--lora_r', type=int,default=1)
    # parser.add_argument('--prefix', type=str,default='Language: ')
    # parser.add_argument('--soft_token_num', type=int,default=4)
    parser.add_argument('--epoch', type=int,default=50)
    parser.add_argument('--num_warmup_steps', type=int,default=0)
    parser.add_argument('--n_shot', type=int,default=None)
    parser.add_argument('--train_bs', type=int,default=4)
    parser.add_argument('--eval_bs', type=int,default=100)
    parser.add_argument('--eval_steps', type=int,default=4)
    parser.add_argument('--text_lr', type=float,default=1e-3)
    # parser.add_argument('--image_lr', type=float,default=1e-4)
    parser.add_argument('--weight_decay', type=float,default=0)
    parser.add_argument('--seed', type=int,default=1)
    # parser.add_argument('--loss_strategy', type=str,choices=['cosine_overall','cosine_mse','fairness',"cosine_only","mix_triple","mutual",'jsd','jsd3'],default='cosine_only')
    # parser.add_argument('--train_target', type=str,choices=['independent','translation'],default='independent')
    parser.add_argument('--output_basedir', type=str,default='./prompt/outputs')
    # parser.add_argument('--mutual_coef', type=float,default=0.1)
    args = parser.parse_args()
    return args

def prepare(args):
    times = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time()))
    args.output_dir = os.path.join(args.output_basedir,f"{args.if_weight}_{args.delta_tuning}_{args.n_shot}_{args.language_list}_t{args.text_lr}_{times}")
    if not os.path.exists(args.output_basedir):
        os.mkdir(args.output_basedir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        logger.error('dir exists')
        exit()
    add_filehandler(os.path.join(args.output_dir,'out.log'))
    
    set_seed(args.seed)
    
    logger.info(args)

if __name__=='__main__':

    
    args = get_args()

    prepare(args)
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(args.device)
    
    image_model, preprocess = clip.load("ViT-L/14", device=args.device)
    text_model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
        text_model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_name)

    text_model = text_model.to(args.device)
    image_model = image_model.to(args.device)

    ##
    # train data
    independent_train_images, independent_train_text = load_file_from_task(task_idx=1,n_obs=args.n_shot,split='train', image_preprocessor=preprocess)
    train_data = [independent_train_images]+[independent_train_text[l] for l in args.language_list]
    train_dataloader = torch.utils.data.DataLoader([i for i in zip(*train_data)] , batch_size=args.train_bs, shuffle=True, num_workers=0)
    # if args.joint_lang_train:
    #     train_dataloader = torch.utils.data.DataLoader([(img, txt_en, txt_de, txt_fr, txt_cs) for img, txt_en, txt_de, txt_fr, txt_cs in zip(independent_train_images, independent_train_text['en'],independent_train_text['de'],independent_train_text['fr'],independent_train_text['cs'])] , batch_size=args.train_bs, shuffle=True, num_workers=0)
    # else:
    #     train_dataloader = torch.utils.data.DataLoader([(img, txt) for img, txt in zip(independent_train_images, independent_train_text[args.language])] , batch_size=args.train_bs, shuffle=True, num_workers=0)

    logger.info(f"train len: {len(independent_train_images)}")

    ##
    # independent_val data
    independent_val_images, independent_val_texts = load_file_from_task(task_idx=1,n_obs=args.n_shot,split='val', image_preprocessor=preprocess)
    independent_val_images_dataloader = torch.utils.data.DataLoader(independent_val_images, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    independent_val_texts_dataloader = {}
    for l in independent_val_texts:
        independent_val_texts_dataloader[l] = torch.utils.data.DataLoader(independent_val_texts[l], batch_size=args.eval_bs, shuffle=False, num_workers=0)
    logger.info(f"val len: {len(independent_val_images)}")

    ##
    # independent_test data
    independent_test_images_1, independent_test_texts_1 = load_file_from_task(task_idx=1,n_obs=None,split='test_2016_flickr', image_preprocessor=preprocess)
    independent_test_images_dataloader_1 = torch.utils.data.DataLoader(independent_test_images_1, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    independent_test_texts_dataloader_1 = {}
    for l in independent_test_texts_1:
        independent_test_texts_dataloader_1[l] = torch.utils.data.DataLoader(independent_test_texts_1[l], batch_size=args.eval_bs, shuffle=False, num_workers=0)

    independent_test_images_2, independent_test_texts_2 = load_file_from_task(task_idx=1,n_obs=None,split='c', image_preprocessor=preprocess)
    independent_test_images_dataloader_2 = torch.utils.data.DataLoader(independent_test_images_2, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    independent_test_texts_dataloader_2 = {}
    for l in independent_test_texts_2:
        independent_test_texts_dataloader_2[l] = torch.utils.data.DataLoader(independent_test_texts_2[l], batch_size=args.eval_bs, shuffle=False, num_workers=0)
    logger.info(f"test_2016_flickr len: {len(independent_test_images_1)}")
    logger.info(f"XTD len: {len(independent_test_images_2)}")
    # translation_val_images_dataloader,translation_val_texts_dataloader = None,None

    # if args.delta_tuning:
    #     from conditional_compacter import ConditionalCompacterModel
    #     image_delta_config = {
    #         "modified_modules": ['mlp', 'attn'],
    #         "unfrozen_modules": ['deltas', 'ln_1', 'ln_2'],
    #         "language_embeding_dim": 768,
    #         "dtype": image_model.dtype,
    #         "device": args.device
    #     }
    #     image_delta_model = ConditionalCompacterModel(
    #         image_model.visual, **image_delta_config)
    #     image_delta_model.freeze_module(set_state_dict=True)
    #     image_delta_model.log(
    #         delta_ratio=True, trainable_ratio=True, visualization=True)
    #     image_model.to(args.device)

    # else:
    #     image_delta_model = None

    # image_unfreezed_modules = [
    #         p for p in image_model.visual.parameters() if p.requires_grad]
    # logger.info(
    #     f"image model: {(sum([p.numel() for p in image_unfreezed_modules])/1024**2)}/{(sum([p.numel() for p in image_model.visual.parameters()])/1024**2)}")
    # logger.info(
    #     f"image model radio: {(sum([p.numel() for p in image_unfreezed_modules])/1024**2)/(sum([p.numel() for p in image_model.visual.parameters()])/1024**2)}")
    
    # image_optimizer = torch.optim.AdamW(image_unfreezed_modules, lr=args.image_lr,weight_decay=args.weight_decay)
    # image_lr_schedule = transformers.get_cosine_schedule_with_warmup(optimizer=image_optimizer,num_warmup_steps=args.num_warmup_steps,num_training_steps=len(train_dataloader)*args.epoch)
    image_delta_model = None
    image_optimizer = None
    image_lr_schedule = None
    image_model.requires_grad_(False)

    # =========================================================================
    # text model
    if args.delta_tuning:
        from opendelta import AutoDeltaConfig,AutoDeltaModel
        delta_config = {
            "delta_type":"lora",
            "backbone_model":'m-clip',
            "lora_r": args.lora_r,
            "modified_modules":['query','value','key','attention.output.dense'],
            "unfrozen_modules":['deltas','LinearTransformation','pooler','output.LayerNorm'],
        }
        config = AutoDeltaConfig.from_dict(delta_config)
        text_delta_model = AutoDeltaModel.from_config(config, text_model.transformer)
        text_delta_model.freeze_module(set_state_dict = True)
        text_delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True, printfile=args.output_dir)

        
    else:
        text_delta_model = None
    text_unfreezed_modules = [p for p in text_model.parameters() if p.requires_grad]
    logger.info(
        f"text model: {(sum([p.numel() for p in text_unfreezed_modules])/1024**2)}/{(sum([p.numel() for p in text_model.parameters()])/1024**2)}")
    logger.info(
        f"text model radio: {(sum([p.numel() for p in text_unfreezed_modules])/1024**2)/(sum([p.numel() for p in text_model.parameters()])/1024**2)}")
    text_optimizer = torch.optim.AdamW(text_unfreezed_modules, lr=args.text_lr,weight_decay=args.weight_decay)
    text_lr_schedule = transformers.get_cosine_schedule_with_warmup(optimizer=text_optimizer,num_warmup_steps=args.num_warmup_steps,num_training_steps=len(train_dataloader)*args.epoch)

    # train
    train(args=args,
        text_model=text_model,tokenizer=tokenizer,
        image_model=image_model,
        text_delta_model=text_delta_model,
        image_delta_model=image_delta_model,
        text_optimizer=text_optimizer,text_lr_schedule=text_lr_schedule,
        image_optimizer=image_optimizer,image_lr_schedule=image_lr_schedule,
        train_dataloader=train_dataloader,
        independent_val_images_dataloader=independent_val_images_dataloader,
        independent_val_texts_dataloader=independent_val_texts_dataloader
        )

    # final test
    #save final
    torch.save(get_model_state_dict_requires_grad(text_model), os.path.join(args.output_dir,'text_model_final.ckpt'))
    # torch.save(image_model.visual.state_dict(), os.path.join(args.output_dir,'image_model_final.ckpt'))

    estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_1,independent_test_texts_dataloader_1,desciption=f'test_2016_flickr {args.n_shot}-shot final',count_lang=args.language_list,args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_2,independent_test_texts_dataloader_2,desciption=f'XTD {args.n_shot}-shot final',count_lang=args.language_list,args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    # estimate(text_model,tokenizer,image_model,translation_test_images_dataloader,translation_test_texts_dataloader,device=device,desciption='translation_test final',count_lang=args.language)

    #load best model
    text_model.load_state_dict(torch.load(os.path.join(args.output_dir,'text_model_best.ckpt')),strict=False)
    # image_model.visual.load_state_dict(torch.load(os.path.join(args.output_dir,'image_model_best.ckpt')),strict=False)
    result_1 = estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_1,independent_test_texts_dataloader_1,desciption=f'test_2016_flickr {args.n_shot}-shot best',count_lang=args.language_list,args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    result_2 = estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_2,independent_test_texts_dataloader_2,desciption=f'XTD {args.n_shot}-shot best',count_lang=args.language_list,args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    # img2text_overall_t, text2img_overall_t = estimate(text_model,tokenizer,image_model,translation_test_images_dataloader,translation_test_texts_dataloader,device=device,desciption='translation_test best',count_lang=args.language)

    result = {'dir':args.output_dir,
        'test_2016_flickr':result_1,
        'XTD':result_2,
    }
    with open("./collect_result.jsonl", 'a') as fout:
            string = json.dumps(result, indent=4, sort_keys=False)
            fout.write(string+"\n")
            
    if torch.cuda.is_available():
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 3)
        logger.info(f"Memory utilization {peak_memory} GB")
