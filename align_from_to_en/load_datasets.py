from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import os
import datasets
FILE_PARH={
    'train':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/train.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/raw/train',
        'languages':['en','de','fr','cs']
    },
    'train_trans_from_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/train.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/translated_from_en/train',
        'languages':['en','de','fr','cs']
    },
    'train_trans_to_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/train.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/translated_to_en/train',
        'languages':['en','de','fr','cs']
    },
    'val':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/val.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/raw/val',
        'languages':['en','de','fr','cs']
    },
    'val_trans_from_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/val.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/translated_from_en/val',
        'languages':['en','de','fr','cs']
    },
    'val_trans_to_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/val.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/translated_to_en/val',
        'languages':['en','de','fr','cs']
    },
    'test_2016_flickr':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/test_2016_flickr.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/raw/test_2016_flickr',
        'languages':['en','de','fr','cs']
    },
    'test_2016_flickr_trans_from_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/test_2016_flickr.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/translated_from_en/test_2016_flickr',
        'languages':['en','de','fr','cs']
    },
    'test_2016_flickr_trans_to_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/test_2016_flickr.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/translated_to_en/test_2016_flickr',
        'languages':['en','de','fr','cs']
    },
    # 'test_2017_flickr':{
    #     'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/test_2017_flickr.txt',
    #     'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/raw/test_2017_flickr',
    #     'languages':['en','de','fr']
    # },
    # 'test_2017_mscoco':{
    #     'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/test_2017_mscoco.txt',
    #     'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/raw/test_2017_mscoco',
    #     'languages':['en','de','fr']
    # },
    # 'test_2018_flickr':{
    #     'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/test_2017_mscoco.txt',
    #     'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/raw/test_2018_flickr',
    #     'languages':['en','de','fr','cs']
    # },
    'XTD10':{
        'image':'/data1/anonymous/dir3/datasets/Cross-lingual-Test-Dataset-XTD10/all/test_image_names.txt',
        'text':'/data1/anonymous/dir3/datasets/Cross-lingual-Test-Dataset-XTD10/all/test_1kcaptions_',
        'languages':['en','de','fr','es','it','jp','ko','pl','ru','tr','zh']
    },
    'XTD10_trans_from_en':{
        'image':'/data1/anonymous/dir3/datasets/Cross-lingual-Test-Dataset-XTD10/all/test_image_names.txt',
        'text':'/data1/anonymous/dir3/datasets/Cross-lingual-Test-Dataset-XTD10/translated_from_en/',
        'languages':['en','de','fr','es','it','jp','ko','pl','ru','tr','zh']
    },
    'XTD10_trans_to_en':{
        'image':'/data1/anonymous/dir3/datasets/Cross-lingual-Test-Dataset-XTD10/all/test_image_names.txt',
        'text':'/data1/anonymous/dir3/datasets/Cross-lingual-Test-Dataset-XTD10/translated_to_en/',
        'languages':['en','de','fr','es','it','jp','ko','pl','ru','tr','zh']
    },
}


FILE_PARH_TASK2={
    'train':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/train_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/raw/train',
        'languages':['en','de']
    },
    'train_trans_from_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/train_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/translated_from_en/train',
        'languages':['en','de']
    },
    'train_trans_to_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/train_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/translated_to_en/train',
        'languages':['en','de']
    },
    'val':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/val_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/raw/val',
        'languages':['en','de']
    },
    'val_trans_from_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/val_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/translated_from_en/val',
        'languages':['en','de']
    },
    'val_trans_to_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/val_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/translated_to_en/val',
        'languages':['en','de']
    },
    'test_2016_flickr':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/test_2016_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/raw/test_2016',
        'languages':['en','de']
    },
    'test_2016_flickr_trans_from_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/test_2016_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/translated_from_en/test_2016',
        'languages':['en','de']
    },
    'test_2016_flickr_trans_to_en':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/test_2016_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/translated_to_en/test_2016',
        'languages':['en','de']
    }
}
FILE_PARH_TASK2_TRANSLATION={
    'train':{
        'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/image_splits/train_images.txt',
        'text':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task2/raw/train',
        'languages':['en2de','de2en']
    }
}
# FILE_PARH_WIT={
#     'de':{
#         # 'image':'/data1/anonymous/dir3/datasets/multi30k/multi30k-dataset/data/task1/image_splits/train.txt',
#         'text':'/data1/anonymous/dir3/datasets/wit_base/translation',
#         'languages':'de',
#         'target_languages':['en']
#     }
# }
def load_file(split='train',image_preprocessor=None):
    path=FILE_PARH[split]
    if ('train' in split) and os.path.exists('/data1/anonymous/dir3/cache/train_task1.pt'):
        images = torch.load('/data1/anonymous/dir3/cache/train_task1.pt')
    elif ('val' in split) and os.path.exists('/data1/anonymous/dir3/cache/val_task1.pt'):
        images = torch.load('/data1/anonymous/dir3/cache/val_task1.pt')
    elif ('test_2016_flickr' in split) and os.path.exists('/data1/anonymous/dir3/cache/test_2016_flickr_task1.pt'):
        images = torch.load('/data1/anonymous/dir3/cache/test_2016_flickr_task1.pt')
    elif ('XTD' in split) and os.path.exists('/data1/anonymous/dir3/cache/XTD.pt'):
        images = torch.load('/data1/anonymous/dir3/cache/XTD.pt')
    else:
        images = []
        with open(path['image'],'r') as image_file:
            context = image_file.read()
            image_names = context.strip().split('\n')
            for name in tqdm(image_names):
                if split=='test_2017_mscoco' or split=='XTD10' or split=='XTD10_trans_from_en' or split=='XTD10_trans_to_en':
                    folder = name.split('_')[1]
                    true_name = name.split('#')[0]
                    image=Image.open(f'/data1/anonymous/dir3/datasets/coco2014/{folder}/{true_name}')
                else:
                    image = Image.open(f'/data1/anonymous/dir3/datasets/multi30k/flickr30k-images/{name}')
                if image_preprocessor is not None:
                    image = image_preprocessor(image).unsqueeze(0)
                    # image = torch.Tensor(np.array(image)).unsqueeze(0)
                images.append(image)
        if image_preprocessor is not None:
            images = torch.cat(images,dim=0)
        if 'train' in split:
            torch.save(images,'/data1/anonymous/dir3/cache/train_task1.pt')
        elif 'val' in split:
            torch.save(images,'/data1/anonymous/dir3/cache/val_task1.pt')
        elif 'test_2016_flickr' in split:
            torch.save(images,'/data1/anonymous/dir3/cache/test_2016_flickr_task1.pt')
        elif 'XTD' in split:
            torch.save(images,'/data1/anonymous/dir3/cache/XTD.pt')
            
    texts = {}
    for l in path['languages']:
        if split=='XTD10' or split=='XTD10_trans_from_en' or split=='XTD10_trans_to_en':
            text_path = f"{path['text']}{l}.txt"
        else:
            text_path = f"{path['text']}.{l}"
        with open(text_path,'r') as text_file:
            context = text_file.read()
            sentences = context.strip().split('\n')
            sentences = [s.strip() for s in sentences]
            texts[l]=sentences

    return images,texts


def load_file_task2(split='train',idx=1,image_preprocessor=None):
    path=FILE_PARH_TASK2[split]
    if ('train' in split) and os.path.exists('/data1/anonymous/dir3/cache/train_task2.pt'):
        images = torch.load('/data1/anonymous/dir3/cache/train_task2.pt')
    elif ('val' in split) and os.path.exists('/data1/anonymous/dir3/cache/val_task2.pt'):
        images = torch.load('/data1/anonymous/dir3/cache/val_task2.pt')
    elif ('test_2016_flickr' in split) and os.path.exists('/data1/anonymous/dir3/cache/test_2016_flickr_task2.pt'):
        images = torch.load('/data1/anonymous/dir3/cache/test_2016_flickr_task2.pt')
    else:
        images = []
        with open(path['image'],'r') as image_file:
            context = image_file.read()
            image_names = context.strip().split('\n')
            for name in tqdm(image_names):
                
                image = Image.open(f'/data1/anonymous/dir3/datasets/multi30k/flickr30k-images/{name}')
                if image_preprocessor is not None:
                    image = image_preprocessor(image).unsqueeze(0)
                    # image = torch.Tensor(np.array(image)).unsqueeze(0)
                images.append(image)
        if image_preprocessor is not None:
            images = torch.cat(images,dim=0)

        if 'train' in split:
            torch.save(images,'/data1/anonymous/dir3/cache/train_task2.pt')
        elif 'val' in split:
            torch.save(images,'/data1/anonymous/dir3/cache/val_task2.pt')
        elif 'test_2016_flickr' in split:
            torch.save(images,'/data1/anonymous/dir3/cache/test_2016_flickr_task2.pt')
    texts = {}
    for l in path['languages']:
        text_path = f"{path['text']}.{idx}.{l}"
        with open(text_path,'r') as text_file:
            context = text_file.read()
            sentences = context.strip().split('\n')
            sentences = [s.strip() for s in sentences]
            texts[l]=sentences

    return images,texts

def load_file_task2_translation(split='train',idx=1,image_preprocessor=None):
    path=FILE_PARH_TASK2_TRANSLATION[split]

    texts = {}
    for l in path['languages']:
        text_path = f"{path['text']}.{idx}.{l}"
        with open(text_path,'r') as text_file:
            context = text_file.read()
            sentences = context.strip().split('\n')
            sentences = [s.strip() for s in sentences]
            texts[l]=sentences

    return None,texts



def load_file_from_task(task_idx=1,n_obs=None,**kwargs):

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


    if task_idx == 1:
        images,texts = load_file(**kwargs)

    elif task_idx == 2:
        images,texts = load_file_task2(**kwargs)
    elif task_idx == 3:
        images,texts = load_file_task2_translation(**kwargs)
    elif task_idx == 4:
        # images,texts = load_file(**kwargs)
        images,texts = load_file(split='XTD10_trans_from_en',image_preprocessor=kwargs['image_preprocessor'])
        assert n_obs is not None
        indices = subsample(images,2*n_obs,seed=42)
        total_num = len(images)
        images_train = images[indices[:n_obs]]
        images_val = images[indices[n_obs:2*n_obs]]
        test_indices = list(set(range(total_num)) - set(indices))
        images_test = images[test_indices]
        texts_train = {}
        texts_val = {}
        texts_test = {}
        for l in texts:
            texts_train[l] = [texts[l][i] for i in indices[:n_obs]]
            texts_val[l] = [texts[l][i] for i in indices[n_obs:2*n_obs]]
            texts_test[l] = [texts[l][i] for i in test_indices]
        return images_train,images_val,images_test,texts_train,texts_val,texts_test



    if n_obs is not None:
        indices = subsample(images,n_obs,seed=42)
        if images is not None:
            images = images[indices]
        for l in texts:
            texts[l] = [texts[l][i] for i in indices]

    

    return images,texts


    