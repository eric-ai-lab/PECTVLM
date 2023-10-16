import torch
from tqdm import tqdm
import itertools
import os
import clip
# import sys;sys.path.append("/data1/anonymous/dir2/naive_finetuning")
from mylogging import add_filehandler, get_logger
from util import recall_k,get_model_state_dict_requires_grad

logger = get_logger(__name__)

LANGUAGE_SPELLING = {
    'en':'English',
    'de':'German',
    'fr':'franch',
    'cs':'Czech',
    'es':"Spanish",
    'it':"Italian",
    'jp':"Japanese",
    'ko':"Korean",
    'pl':"Polish",
    'ru':"Russian",
    'tr':"Turkish",
    'zh':"Chinese",
}
@torch.no_grad()
def estimate_monolingual(text_model,tokenizer,image_model,image_dataloader,text_dataloader,desciption='',count_lang=None,args=None,text_delta_model=None,image_delta_model=None):

    text_model.eval()
    image_model.eval()

    images_features_list = []
    for i in tqdm(image_dataloader):
        images_features = image_model.encode_image(i.to(args.device))
        images_features_list.append(images_features.cpu())
    images_features_all = torch.cat(images_features_list,dim=0)
    # logger.info(images_features_all.shape)

    texts_features_all = {}
    for l in text_dataloader:

        texts_features_all[l]=[]
        single_texts_dataloader = text_dataloader[l]
        for t in tqdm(single_texts_dataloader):
            # for idx,st in enumerate(t):
            #     if len(st) > 70:
            #         t[idx] = st[:70]
            t = clip.tokenize(t,77,True).to(args.device)
            texts_features = text_model.encode_text(t)
            texts_features_all[l].append(texts_features.cpu())
        texts_features_all[l] = torch.cat(texts_features_all[l],dim=0)
    # logger.info(texts_features_all.shape)

    num = len(images_features_all)

    scaled_cos = {}
    images_features_all /= images_features_all.norm(dim=-1, keepdim=True)
    for l in texts_features_all:
        texts_features_all[l] /= texts_features_all[l].norm(dim=-1, keepdim=True)
        scaled_cos[l] = (100.0 * texts_features_all[l].float() @ images_features_all.float().T)

    recall_mean_overall = 0
    labels = torch.Tensor(list(range(num))).to(images_features_all.device)
    logger.info(desciption)
    total_count_num = 0
    for l in scaled_cos:
        recall_mean = recall_k(labels,scaled_cos[l].T,[1],[1])
        logger.info(f'{l} | recall_mean:{recall_mean:.4f} | count: {l in count_lang}')
        if l in count_lang:
            recall_mean_overall+=float(recall_mean)
            total_count_num += 1

    # return img2text_overall/total_count_num, text2img_overall/total_count_num
    if total_count_num==0:
        return 0
    return round(recall_mean_overall/total_count_num,4)


@torch.no_grad()
def estimate(text_model,tokenizer,image_model,image_dataloader,text_dataloader,desciption='',count_lang=None,args=None,text_delta_model=None,image_delta_model=None):

    text_model.eval()
    image_model.eval()

    images_features_list = []
    for i in tqdm(image_dataloader):
        images_features = image_model.encode_image(i.to(args.device))
        images_features_list.append(images_features.cpu())
    images_features_all = torch.cat(images_features_list,dim=0)
    # logger.info(images_features_all.shape)

    texts_features_all = {}
    for l in text_dataloader:

        texts_features_all[l]=[]
        single_texts_dataloader = text_dataloader[l]
        for t in tqdm(single_texts_dataloader):
            texts_features = text_model.forward(t, tokenizer)
            texts_features_all[l].append(texts_features.cpu())
        texts_features_all[l] = torch.cat(texts_features_all[l],dim=0)
    # logger.info(texts_features_all.shape)

    num = len(images_features_all)

    scaled_cos = {}
    images_features_all /= images_features_all.norm(dim=-1, keepdim=True)
    for l in texts_features_all:
        texts_features_all[l] /= texts_features_all[l].norm(dim=-1, keepdim=True)
        scaled_cos[l] = (100.0 * texts_features_all[l] @ images_features_all.float().T)

    recall_mean_overall = 0
    labels = torch.Tensor(list(range(num))).to(images_features_all.device)
    logger.info(desciption)
    total_count_num = 0
    for l in scaled_cos:
        recall_mean = recall_k(labels,scaled_cos[l].T,[1],[1])
        logger.info(f'{l} | recall_mean:{recall_mean:.4f} | count: {l in count_lang}')
        if l in count_lang:
            recall_mean_overall+=float(recall_mean)
            total_count_num += 1

    # return img2text_overall/total_count_num, text2img_overall/total_count_num
    if total_count_num==0:
        return 0
    return round(recall_mean_overall/total_count_num,4)



def train(args,
        text_model,tokenizer,
        image_model,
        text_delta_model,
        image_delta_model,
        text_optimizer,text_lr_schedule,
        image_optimizer,image_lr_schedule,
        train_dataloader,
        images_val_dataloader,
        texts_val_dataloader
        ):


    # logger.info(f"loss_strategy: {args.loss_strategy}")
    total_steps = len(train_dataloader)*args.epoch
    logger.info(f"total steps: {total_steps}")
    global_steps = 0
    max_score = -1
    for e in range(args.epoch):
        for data in tqdm(train_dataloader):
            text_model.train()
            image_model.train()
            global_steps+=1
            

            # img_s,en_s,de_s,fr_s,cs_s = data
            # images,ind_text_en,ind_text_tgt,from_text,to_text = data
            images = data[0]
            texts_list = data[1:]
            # images = data[0]
            # texts = data[1:]
            batch_size = len(images)
            labels = torch.Tensor(list(range(batch_size))).long().to(args.device)

            with torch.no_grad():
                image_feature = image_model.encode_image(images.to(args.device)).float()
                image_feature_for_loss = image_feature/image_feature.norm(dim=-1, keepdim=True)
            if args.strategy == 'preliminary':
                loss = 0
                for texts in texts_list:
                    text_feature_norm = text_model.forward(list(texts), tokenizer)
                    text_feature_norm = text_feature_norm/text_feature_norm.norm(dim=-1, keepdim=True)

                
                    scaled_cos = (100.0 * image_feature_for_loss @ text_feature_norm.T)

                    loss += torch.nn.functional.cross_entropy(scaled_cos, labels)
                    loss += torch.nn.functional.cross_entropy(scaled_cos.T, labels)
                    scaled_cos=0
                    text_feature_norm=0
                    # torch.cuda.empty_cache()


                    loss = loss/2

                loss = loss/len(texts_list)

            elif args.strategy=='method3' or args.strategy=='baseline2':
                to_text_feature = text_model.forward(list(to_text), tokenizer)
                to_text_feature_norm = to_text_feature/to_text_feature.norm(dim=-1, keepdim=True)
                
                scaled_cos = (100.0 * image_feature_for_loss @ to_text_feature_norm.T)

                loss_i = torch.nn.functional.cross_entropy(scaled_cos, labels)
                loss_t = torch.nn.functional.cross_entropy(scaled_cos.T, labels)

                loss = (loss_i + loss_t)/2

                if args.strategy=='method3':
                    if args.text_loss_strategy=='contrastive':
                        ind_text_en_feature = text_model.forward(list(ind_text_en), tokenizer)
                        ind_text_en_feature_norm = ind_text_en_feature/ind_text_en_feature.norm(dim=-1, keepdim=True)

                        text_scaled_cos = (100.0 * ind_text_en_feature_norm @ to_text_feature_norm.T)
                        loss_a = torch.nn.functional.cross_entropy(text_scaled_cos, labels)
                        loss_b = torch.nn.functional.cross_entropy(text_scaled_cos.T, labels)

                        loss += args.text_loss_ratio * (loss_a + loss_b)/2
                    elif args.text_loss_strategy=='mse':
                        ind_text_en_feature = text_model.forward(list(ind_text_en), tokenizer)
                        # ind_text_en_feature_norm = ind_text_en_feature/ind_text_en_feature.norm(dim=-1, keepdim=True)
                        loss_fct_text_pair = torch.nn.MSELoss()
                        mse_loss = loss_fct_text_pair(to_text_feature,ind_text_en_feature)
                        loss += args.text_loss_ratio * mse_loss

                    elif args.text_loss_strategy=='contrastive_with_image':
                        ind_text_en_feature = text_model.forward(list(ind_text_en), tokenizer)
                        ind_text_en_feature_norm = ind_text_en_feature/ind_text_en_feature.norm(dim=-1, keepdim=True)

                        text_scaled_cos = (100.0 * image_feature_for_loss @ ind_text_en_feature_norm.T)
                        loss_a = torch.nn.functional.cross_entropy(text_scaled_cos, labels)
                        loss_b = torch.nn.functional.cross_entropy(text_scaled_cos.T, labels)

                        loss += args.text_loss_ratio * (loss_a + loss_b)/2
                    else:
                        raise NotImplementedError
            
            elif args.strategy=='method2' or args.strategy=='baseline1' or args.strategy=='method1':
                ind_text_tgt_feature = text_model.forward(list(ind_text_tgt), tokenizer)
                ind_text_tgt_feature_norm = ind_text_tgt_feature/ind_text_tgt_feature.norm(dim=-1, keepdim=True)
                
                scaled_cos = (100.0 * image_feature_for_loss @ ind_text_tgt_feature_norm.T)

                loss_i = torch.nn.functional.cross_entropy(scaled_cos, labels)
                loss_t = torch.nn.functional.cross_entropy(scaled_cos.T, labels)

                loss = (loss_i + loss_t)/2

                if args.strategy=='method1':
                    if args.text_loss_strategy=='contrastive':
                        ind_text_en_feature = text_model.forward(list(ind_text_en), tokenizer)
                        ind_text_en_feature_norm = ind_text_en_feature/ind_text_en_feature.norm(dim=-1, keepdim=True)

                        text_scaled_cos = (100.0 * ind_text_en_feature_norm @ ind_text_tgt_feature_norm.T)
                        loss_a = torch.nn.functional.cross_entropy(text_scaled_cos, labels)
                        loss_b = torch.nn.functional.cross_entropy(text_scaled_cos.T, labels)

                        loss += args.text_loss_ratio * (loss_a + loss_b)/2
                    elif args.text_loss_strategy=='mse':
                        ind_text_en_feature = text_model.forward(list(ind_text_en), tokenizer)

                        loss_fct_text_pair = torch.nn.MSELoss()
                        mse_loss = loss_fct_text_pair(ind_text_tgt_feature,ind_text_en_feature)
                        loss += args.text_loss_ratio * mse_loss
                    elif args.text_loss_strategy=='contrastive_with_image':
                        ind_text_en_feature = text_model.forward(list(ind_text_en), tokenizer)
                        ind_text_en_feature_norm = ind_text_en_feature/ind_text_en_feature.norm(dim=-1, keepdim=True)

                        text_scaled_cos = (100.0 * image_feature_for_loss @ ind_text_en_feature_norm.T)
                        loss_a = torch.nn.functional.cross_entropy(text_scaled_cos, labels)
                        loss_b = torch.nn.functional.cross_entropy(text_scaled_cos.T, labels)

                        loss += args.text_loss_ratio * (loss_a + loss_b)/2
                    else:
                        raise NotImplementedError
                
                elif args.strategy=='method2':
                    if args.text_loss_strategy=='contrastive':
                        from_text_feature = text_model.forward(list(from_text), tokenizer)
                        from_text_feature_norm = from_text_feature/from_text_feature.norm(dim=-1, keepdim=True)
                        
                        text_scaled_cos = (100.0 * from_text_feature_norm @ ind_text_tgt_feature_norm.T)
                        loss_a = torch.nn.functional.cross_entropy(text_scaled_cos, labels)
                        loss_b = torch.nn.functional.cross_entropy(text_scaled_cos.T, labels)

                        loss += args.text_loss_ratio * (loss_a + loss_b)/2
                    elif args.text_loss_strategy=='mse':
                        from_text_feature = text_model.forward(list(from_text), tokenizer)

                        loss_fct_text_pair = torch.nn.MSELoss()
                        mse_loss = loss_fct_text_pair(ind_text_tgt_feature,from_text_feature)
                        loss += args.text_loss_ratio * mse_loss
                    elif args.text_loss_strategy=='contrastive_with_image':
                        from_text_feature = text_model.forward(list(from_text), tokenizer)
                        from_text_feature_norm = from_text_feature/from_text_feature.norm(dim=-1, keepdim=True)
                        
                        text_scaled_cos = (100.0 * image_feature_for_loss @ from_text_feature_norm.T)
                        loss_a = torch.nn.functional.cross_entropy(text_scaled_cos, labels)
                        loss_b = torch.nn.functional.cross_entropy(text_scaled_cos.T, labels)

                        loss += args.text_loss_ratio * (loss_a + loss_b)/2
                    else:
                        raise NotImplementedError



            loss.backward()
            text_optimizer.step()
            # image_optimizer.step()
            text_lr_schedule.step()
            # image_lr_schedule.step()
            
            text_optimizer.zero_grad()
            # image_optimizer.zero_grad()


            if (global_steps % args.eval_steps) == 0:
                logger.info('eval')
                if args.strategy=='preliminary':
                    recall_overall_i =  estimate(text_model,tokenizer,image_model,images_val_dataloader,texts_val_dataloader[1],desciption=f'independent_val epoch-{e} step-{global_steps}',count_lang=args.language_list,args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
                elif args.strategy=='method3' or args.strategy=='baseline2':
                    recall_overall_i =  estimate(text_model,tokenizer,image_model,images_val_dataloader,texts_val_dataloader[0],desciption=f'independent_val epoch-{e} step-{global_steps}',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
                elif args.strategy=='method2' or args.strategy=='baseline1' or args.strategy=='method1':
                    recall_overall_i =  estimate(text_model,tokenizer,image_model,images_val_dataloader,texts_val_dataloader[1],desciption=f'independent_val epoch-{e} step-{global_steps}',count_lang=[args.language],args=args,text_delta_model=text_delta_model,image_delta_model=image_delta_model)
                
                now_score = recall_overall_i
                logger.info(f'score: {now_score}, text_lr: {text_lr_schedule.get_last_lr()}')
                if now_score >= max_score:
                    max_score = now_score
                    torch.save(get_model_state_dict_requires_grad(text_model), os.path.join(args.output_dir,'text_model_best.ckpt'))
                    # torch.save(image_model.visual.state_dict(), os.path.join(args.output_dir,'image_model_best.ckpt'))
                    logger.info('best text model saved')

                