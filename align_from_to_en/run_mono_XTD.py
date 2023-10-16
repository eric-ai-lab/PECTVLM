import sys;sys.path.append("/data1/anonymous/dir2/lib")
from Multilingual_CLIP.multilingual_clip import pt_multilingual_clip
from trainer import train, estimate, estimate_monolingual
from load_datasets import load_file_from_task,load_file,load_file_task2
import clip
import open_clip
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
    parser.add_argument('--language', type=str,choices=['en','de','fr','es','it','jp','ko','pl','ru','tr','zh','cs'],default='en')
    
    # parser.add_argument('--language_list', nargs='+')
    # parser.add_argument('--language2', type=str,choices=['en','de','fr','cs'],default='de')
    # parser.add_argument('--tolanguage', type=str,choices=['en','de'])
    # parser.add_argument('--joint_lang_train', action='store_true')

    parser.add_argument('--delta_tuning', action='store_true')
    parser.add_argument('--delta_type', type=str,choices=['prompt','lora','adapter','compacter'],default='lora')
    parser.add_argument('--lora_r', type=int,default=8)
    parser.add_argument('--adapter_bottleneck_dim', type=int,default=24)
    parser.add_argument('--compacter_reduction_factor', type=int,default=16)
    parser.add_argument('--prefix', type=str,default='Language: ')
    parser.add_argument('--soft_token_num', type=int,default=4)

    parser.add_argument('--epoch', type=int,default=50)
    parser.add_argument('--num_warmup_steps', type=int,default=0)
    parser.add_argument('--n_shot', type=int,default=None)
    parser.add_argument('--train_bs', type=int,default=4)
    parser.add_argument('--eval_bs', type=int,default=200)
    parser.add_argument('--eval_steps', type=int,default=4)
    parser.add_argument('--text_lr', type=float,default=1e-3)
    # parser.add_argument('--image_lr', type=float,default=1e-4)
    parser.add_argument('--weight_decay', type=float,default=0)
    parser.add_argument('--seed', type=int,default=1)
    parser.add_argument('--split_seed', type=int,default=42)
    parser.add_argument('--strategy', type=str,choices=['method1','method2','method3','baseline1','baseline2'])
    parser.add_argument('--text_loss_strategy', type=str,choices=['cosine_mse','mse','contrastive'],default='mse')
    parser.add_argument('--text_loss_ratio', type=float,default=1.0)
    parser.add_argument('--resume', type=str,default=None)
    # parser.add_argument('--train_target', type=str,choices=['independent','translation'],default='independent')
    parser.add_argument('--output_basedir', type=str,default='./outputs')
    # parser.add_argument('--mutual_coef', type=float,default=0.1)
    args = parser.parse_args()
    return args

def prepare(args):
    times = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time()))
    args.output_dir = os.path.join(args.output_basedir,f"{args.language}_{args.delta_tuning}_{args.delta_type}_{args.n_shot}_t{args.text_lr}_r{args.text_loss_ratio}_{args.strategy}_{args.text_loss_strategy}_{times}")
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
def subsample(dataset, n_obs=None, indices=None,seed = 42):
    num_samples = len(dataset)
    if n_obs is not None and n_obs > num_samples:
        n_obs = num_samples

    if indices is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator).tolist()
    indices = indices[:n_obs]
    return indices

class Text_model:
    def __init__(self, model):
        self.model = model
    def forward(self,text,tokenizer):
        return self.model.encode_text(text)

if __name__=='__main__':

    
    args = get_args()
    print(args.prefix)

    prepare(args)
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(args.device)
    
    image_model, preprocess = clip.load("ViT-L/14", device=args.device)
    text_model = image_model
    tokenizer=None
    # image_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    # text_model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
    # text_model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
    # text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
    #     text_model_name)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_name)



    ##
    # train data
    ind_image, ind_text = load_file_from_task(task_idx=1,n_obs=None,split='XTD10', image_preprocessor=preprocess)
    # print(len(ind_image)),print(len(ind_text))
    _, from_text = load_file_from_task(task_idx=1,n_obs=None,split='XTD10_trans_from_en', image_preprocessor=preprocess)
    # print(len(from_text))
    _, to_text = load_file_from_task(task_idx=1,n_obs=None,split='XTD10_trans_to_en', image_preprocessor=preprocess)
    # print(len(to_text))



    generator = torch.Generator()
    generator.manual_seed(args.split_seed)
    indices = torch.randperm(len(ind_image), generator=generator).tolist()
    # [ind_image[i] for i in indices[0:args.n_shot]]
    # ind_image_train, ind_image_val, ind_image_test = ind_image[indices[0:args.n_shot]], ind_image[indices[args.n_shot:2*args.n_shot]], ind_image[indices[2*args.n_shot:]]
    # ind_text_train_en, ind_text_val_en, ind_text_test_en = ind_text['en'][indices[0:args.n_shot]], ind_text['en'][indices[args.n_shot:2*args.n_shot]], ind_text['en'][indices[2*args.n_shot:]]
    # ind_text_train_tgt, ind_text_val_tgt, ind_text_test_tgt = ind_text[args.language][indices[0:args.n_shot]], ind_text[args.language][indices[args.n_shot:2*args.n_shot]], ind_text[args.language][indices[2*args.n_shot:]]
    # from_text_train, from_text_val, from_text_test = from_text[args.language][indices[0:args.n_shot]], from_text[args.language][indices[args.n_shot:2*args.n_shot]], from_text[args.language][indices[2*args.n_shot:]]
    # to_text_train, to_text_val, to_text_test = to_text[args.language][indices[0:args.n_shot]], to_text[args.language][indices[args.n_shot:2*args.n_shot]], to_text[args.language][indices[2*args.n_shot:]]

    ind_image_train, ind_image_val, ind_image_test = [ind_image[i] for i in indices[0:args.n_shot]], [ind_image[i] for i in indices[args.n_shot:2*args.n_shot]], [ind_image[i] for i in indices[2*args.n_shot:]]
    print(len(ind_image_val))
    ind_text_train_en, ind_text_val_en, ind_text_test_en = [ind_text['en'][i] for i in indices[0:args.n_shot]], [ind_text['en'][i] for i in indices[args.n_shot:2*args.n_shot]], [ind_text['en'][i] for i in indices[2*args.n_shot:]]
    ind_text_train_tgt, ind_text_val_tgt, ind_text_test_tgt = [ind_text[args.language][i] for i in indices[0:args.n_shot]], [ind_text[args.language][i] for i in indices[args.n_shot:2*args.n_shot]], [ind_text[args.language][i] for i in indices[2*args.n_shot:]]
    from_text_train, from_text_val, from_text_test = [from_text[args.language][i] for i in indices[0:args.n_shot]], [from_text[args.language][i] for i in indices[args.n_shot:2*args.n_shot]], [from_text[args.language][i] for i in indices[2*args.n_shot:]]
    to_text_train, to_text_val, to_text_test = [to_text[args.language][i] for i in indices[0:args.n_shot]], [to_text[args.language][i] for i in indices[args.n_shot:2*args.n_shot]], [to_text[args.language][i] for i in indices[2*args.n_shot:]]
    
    ind_text_test = {}
    for lang in ind_text:
        print(lang)
        ind_text_test[lang] = [ind_text[lang][i] for i in indices[2*args.n_shot:]]

    to_text_test = {}
    for lang in to_text:
        print(lang)
        to_text_test[lang] = [to_text[lang][i] for i in indices[2*args.n_shot:]]

    # independent_train_images,independent_val_images,independent_test_images_2, independent_train_text,independent_val_texts,independent_test_texts_2 = load_file_from_task(task_idx=4,n_obs=args.n_shot, image_preprocessor=preprocess)
    # train_data = [independent_train_images]+[independent_train_text[l] for l in ['en', args.language]]
    # train_data = [ind_image_train,ind_text_train_en,ind_text_train_tgt,from_text_train,to_text_train]
    # train_dataloader = torch.utils.data.DataLoader([i for i in zip(*train_data)] , batch_size=args.train_bs, shuffle=True, num_workers=0)
    # logger.info(f"train len: {len(ind_image_train)}")

    # ##
    # # independent_val data
    # images_val_dataloader = torch.utils.data.DataLoader(ind_image_val, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # print(len(ind_image_val))
    # ind_texts_val_dataloader = {}
    # ind_texts_val_dataloader[args.language] = torch.utils.data.DataLoader(ind_text_val_tgt, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # # from_texts_val_dataloader = {}
    # # from_texts_val_dataloader[args.language] = torch.utils.data.DataLoader(from_text_val, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # to_texts_val_dataloader = {}
    # to_texts_val_dataloader[args.language] = torch.utils.data.DataLoader(to_text_val, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # logger.info(f"val len: {len(ind_image_val)}")

    ##
    # independent_test data
    images_test_dataloader = torch.utils.data.DataLoader(ind_image_test, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    ind_texts_test_dataloader = {}
    for lang in ind_text_test:
        ind_texts_test_dataloader[lang] = torch.utils.data.DataLoader(ind_text_test[lang], batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # from_texts_test_dataloader = {}
    # from_texts_test_dataloader[args.language] = torch.utils.data.DataLoader(from_text_test, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    to_texts_test_dataloader = {}
    for lang in to_text_test:
        to_texts_test_dataloader[lang] = torch.utils.data.DataLoader(to_text_test[lang], batch_size=args.eval_bs, shuffle=False, num_workers=0)
    logger.info(f"test len: {len(ind_image_test)}")

    # ##
    # # independent_test data
    # # independent_test_images_1, independent_test_texts_1 = load_file_from_task(task_idx=1,n_obs=None,split='test_2016_flickr', image_preprocessor=preprocess)
    # # independent_test_images_dataloader_1 = torch.utils.data.DataLoader(independent_test_images_1, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # # independent_test_texts_dataloader_1 = {}
    # # for l in independent_test_texts_1:
    # #     independent_test_texts_dataloader_1[l] = torch.utils.data.DataLoader(independent_test_texts_1[l], batch_size=args.eval_bs, shuffle=False, num_workers=0)

    # # independent_test_images_2, independent_test_texts_2 = load_file_from_task(task_idx=1,n_obs=None,split='XTD10', image_preprocessor=preprocess)
    # independent_test_images_dataloader_2 = torch.utils.data.DataLoader(independent_test_images_2, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # independent_test_texts_dataloader_2 = {}
    # for l in independent_test_texts_2:
    #     independent_test_texts_dataloader_2[l] = torch.utils.data.DataLoader(independent_test_texts_2[l], batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # # logger.info(f"test_2016_flickr len: {len(independent_test_images_1)}")
    # logger.info(f"XTD len: {len(independent_test_images_2)}")
    # # translation_val_images_dataloader,translation_val_texts_dataloader = None,None
    
    # independent_test_images_3, independent_test_texts_3 = load_file_from_task(task_idx=1,n_obs=None,split='XTD10', image_preprocessor=preprocess)
    # independent_test_images_dataloader_3 = torch.utils.data.DataLoader(independent_test_images_3, batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # independent_test_texts_dataloader_3 = {}
    # for l in independent_test_texts_3:
    #     independent_test_texts_dataloader_3[l] = torch.utils.data.DataLoader(independent_test_texts_3[l], batch_size=args.eval_bs, shuffle=False, num_workers=0)
    # # logger.info(f"test_2016_flickr len: {len(independent_test_images_1)}")
    # logger.info(f"XTD_trans_from_en len: {len(independent_test_images_3)}")
    # estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_3,independent_test_texts_dataloader_3,desciption=f'zero-shot',count_lang=[args.language],args=args,text_delta_model=None,image_delta_model=None)

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
    # text_delta_model 
    # =========================================================================
    # text model
    if args.delta_tuning:
        from opendelta import AutoDeltaConfig,AutoDeltaModel
        if args.delta_type == 'lora':
            
            delta_config = {
                "delta_type":"lora",
                "backbone_model":'m-clip',
                "lora_r": args.lora_r,
                "modified_modules":['query','value'],
                # "unfrozen_modules":['deltas','LinearTransformation','pooler','output.LayerNorm'],
                "unfrozen_modules":['deltas','output.LayerNorm'],
            }
            config = AutoDeltaConfig.from_dict(delta_config)
            text_delta_model = AutoDeltaModel.from_config(config, text_model.transformer)
            text_delta_model.freeze_module(set_state_dict = True)
            text_delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True, printfile=args.output_dir)
        elif args.delta_type == 'prompt':
            from soft_prompt import SoftPromptModel
            delta_config = {
                        "soft_token_num":args.soft_token_num,
                        "std":0.02,
                        "token_init":False,
                        "other_expand_ids":{"attention_mask":1, "token_type_ids":0},
            }
            text_model = text_model.to('cpu')
            text_delta_model = SoftPromptModel(text_model.transformer,tokenizer=tokenizer,prefix=args.prefix,**delta_config)

            text_delta_model.freeze_module(set_state_dict = True)
            for p in text_model.LinearTransformation.parameters():
                p.requires_grad = False
            text_delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True, printfile=args.output_dir)
            # delta_model.save_finetuned(args.output_dir)
            text_model = text_model.to(args.device)
            pass
        elif args.delta_type == 'adapter':
            delta_config = {
                "delta_type":"adapter",
                "backbone_model":'m-clip',
                "bottleneck_dim": args.adapter_bottleneck_dim,
                "modified_modules":['output.dense','attention.output.dense'],
                # "unfrozen_modules":['deltas','LinearTransformation','pooler','output.LayerNorm'],
                "unfrozen_modules":['deltas','output.LayerNorm'],
            }
            config = AutoDeltaConfig.from_dict(delta_config)
            text_delta_model = AutoDeltaModel.from_config(config, text_model.transformer)
            text_delta_model.freeze_module(set_state_dict = True)
            text_delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True, printfile=args.output_dir)

        elif args.delta_type == 'compacter':
            delta_config = {
                "delta_type":"compacter",
                "backbone_model":'m-clip',
                "reduction_factor": args.compacter_reduction_factor,
                "modified_modules":['output.dense','attention.output.dense'],
                # "unfrozen_modules":['deltas','LinearTransformation','pooler','output.LayerNorm'],
                "unfrozen_modules":['deltas','output.LayerNorm'],
            }
            config = AutoDeltaConfig.from_dict(delta_config)
            text_delta_model = AutoDeltaModel.from_config(config, text_model.transformer)
            text_delta_model.freeze_module(set_state_dict = True)
            text_delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True, printfile=args.output_dir)

        
    else:
        text_delta_model = None
    
    # if args.resume is not None:
    #     text_model.load_state_dict(torch.load(args.resume),strict=False)
        
    # text_unfreezed_modules = [p for p in text_model.parameters() if p.requires_grad]
    # logger.info(
    #     f"text model: {(sum([p.numel() for p in text_unfreezed_modules])/1024**2)}/{(sum([p.numel() for p in text_model.parameters()])/1024**2)}")
    # logger.info(
    #     f"text model radio: {(sum([p.numel() for p in text_unfreezed_modules])/1024**2)/(sum([p.numel() for p in text_model.parameters()])/1024**2)}")
    # text_optimizer = torch.optim.Adam(text_unfreezed_modules, lr=args.text_lr,weight_decay=args.weight_decay)
    # text_lr_schedule = transformers.get_cosine_schedule_with_warmup(optimizer=text_optimizer,num_warmup_steps=args.num_warmup_steps,num_training_steps=len(train_dataloader)*args.epoch)
    # estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_2,independent_test_texts_dataloader_2,desciption=f'zero-shot',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)

    # final test
    #save final
    # torch.save(get_model_state_dict_requires_grad(text_model), os.path.join(args.output_dir,'text_model_final.ckpt'))
    # torch.save(image_model.visual.state_dict(), os.path.join(args.output_dir,'image_model_final.ckpt'))

    if args.strategy=='method3' or args.strategy=='baseline2':
        recall_overall_i =  estimate_monolingual(text_model,tokenizer,image_model,images_test_dataloader,to_texts_test_dataloader,desciption=f'XTD {args.n_shot}-shot {args.strategy} final',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    elif args.strategy=='method2' or args.strategy=='baseline1' or args.strategy=='method1':
        recall_overall_i =  estimate_monolingual(text_model,tokenizer,image_model,images_test_dataloader,ind_texts_test_dataloader,desciption=f'XTD {args.n_shot}-shot {args.strategy} final',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)

    # estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_1,independent_test_texts_dataloader_1,desciption=f'test_2016_flickr {args.n_shot}-shot final',count_lang=args.language_list,args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    # estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_2,independent_test_texts_dataloader_2,desciption=f'XTD {args.n_shot}-shot final',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    # estimate(text_model,tokenizer,image_model,translation_test_images_dataloader,translation_test_texts_dataloader,device=device,desciption='translation_test final',count_lang=args.language)

    #load best model
    # text_model.load_state_dict(torch.load(os.path.join(args.output_dir,'text_model_best.ckpt')),strict=False)
    # image_model.visual.load_state_dict(torch.load(os.path.join(args.output_dir,'image_model_best.ckpt')),strict=False)

    # if args.strategy=='method3' or args.strategy=='baseline2':
    #     recall_overall_i =  estimate(text_model,tokenizer,image_model,images_test_dataloader,to_texts_test_dataloader,desciption=f'XTD {args.n_shot}-shot {args.strategy} best',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    # elif args.strategy=='method2' or args.strategy=='baseline1' or args.strategy=='method1':
    #     recall_overall_i =  estimate(text_model,tokenizer,image_model,images_test_dataloader,ind_texts_test_dataloader,desciption=f'XTD {args.n_shot}-shot {args.strategy} best',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    # result_1 = estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_1,independent_test_texts_dataloader_1,desciption=f'test_2016_flickr {args.n_shot}-shot best',count_lang=args.language_list,args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    # result_2 = estimate(text_model,tokenizer,image_model,independent_test_images_dataloader_2,independent_test_texts_dataloader_2,desciption=f'XTD {args.n_shot}-shot best',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
    # img2text_overall_t, text2img_overall_t = estimate(text_model,tokenizer,image_model,translation_test_images_dataloader,translation_test_texts_dataloader,device=device,desciption='translation_test best',count_lang=args.language)

    result = {'dir':args.output_dir,
        # 'test_2016_flickr':result_1,
        'XTD':recall_overall_i,
    }
    with open("./collect_result_prompt.jsonl", 'a') as fout:
        string = json.dumps(result, indent=4, sort_keys=False)
        fout.write(string+"\n")
            
    if torch.cuda.is_available():
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 3)
        logger.info(f"Memory utilization {peak_memory} GB")
