# coding=utf-8
import os
import json
import torch
import os.path
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from transformers import AutoTokenizer
import clip
from TTA_Dataset import COCOCapDatasetForEmbedding, collate
from custom_models import CAP_TTA
from clip_reward import get_reward_model
from search import greedy_search, beam_search, opt_search
import numpy as np
from generate_opt import generate as opt_generate
import itertools
from scene_policy import *
from scene_policy_hw import HW_PPO, HW_Memory
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import pickle
import argparse
from ClipCap import ClipCaptionModel
from utils import compose_discrete_prompts, compose_sg_discrete_prompts
from load_annotations import load_entities_text
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories, \
    compute_image_text_similarity_via_embeddings

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def img_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_patch(images, action_sequence, hw_action):
    batch_size = images.size(0)
    image_size = images.size(2)


    patch_size = torch.floor(hw_action * image_size).int()


    patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int()

    patches = []
    for i in range(batch_size):
        per_patch = images[i, :,
                    (patch_coordinate[i, 0].item()): ((patch_coordinate[i, 0] + patch_size[i, 0]).item()),
                    (patch_coordinate[i, 1].item()): ((patch_coordinate[i, 1] + patch_size[i, 1]).item())]

        patches.append(per_patch.view(1, per_patch.size(0), per_patch.size(1), per_patch.size(2)))

    return torch.cat(patches, 0), patch_size.cpu().numpy(), patch_coordinate.cpu().numpy()


class TxtLogger():
    def __init__(self, log_file) -> None:
        self.log_file = log_file
        with open(self.log_file, "w") as file:
            file.write(log_file + "\n\n")

    def log_id(self, image_id):
        with open(self.log_file, "a") as file:
            file.write(image_id + "\n")

    def log_policy_hw(self, size, corr):
        with open(self.log_file, "a") as file:
            file.write('size:' + str(size[0]) + "\n")
            file.write('corr:' + str(corr[0]) + "\n")

    def log_sample_text(self, sample_text, scores):
        with open(self.log_file, "a") as file:
            for t in sample_text:
                file.write(t + "\n")
            for s in scores:
                file.write(str(round(s, 4)) + "  ")
            file.write("\n")


    def log_final_text(self, final_text):
        with open(self.log_file, "a") as file:
            file.write("final text:" + "\n")
            for t in final_text:
                file.write(t + "\t")
            file.write("\n")
            file.write("\n")


def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice


def tta_nocaps(args,
               inpath,
               entities_text,
               texts_embeddings,
               model,
               tokenizer,
               clip_model,
               preprocess,
               reward_model,
               optimizer,
               optim_state,
               scaler,
               ppo,
               hw_ppo,
               memory,
               hw_memory,
               ori_transform,
               cpu_device
               ):
    device = args.device

    if args.using_image_features:
        with open(inpath, 'rb') as infile:
            annotations = pickle.load(infile)  # [[image_path, image_features, [caption1, caption2, ...]], ...]
    else:
        with open(inpath, 'r') as infile:
            annotations = json.load(infile)  # {image_path: [caption1, caption2, ...]}

    reward_log = os.path.join(args.out_path, f'{args.name_of_datasets}_tta_reward.json')
    text_logger = TxtLogger(reward_log.replace(".json", ".txt"))

    sample_k = reward_model.sample_k

    indomain = []
    neardomain = []
    outdomain = []
    overall = []
    for idx, annotation in enumerate(tqdm(annotations)):

        image_id = annotation['image_id']
        split = annotation['split']
        captions = annotation['caption']
        image_path = args.image_folder + '/' + annotation['file_name']
        image = preprocess(Image.open(image_path)).unsqueeze(dim=0).to(device)
        ori_image = ori_transform(Image.open(image_path)).unsqueeze(dim=0).to(device)

        with torch.no_grad():
            model.eval()
            with torch.cuda.amp.autocast():
                bs_prefix = clip_model.encode_image(image).float()  # [1, 512]
                if args.normalize_prefix: bs_prefix = bs_prefix / bs_prefix.norm(2, -1, keepdim=True)
                global_bs_prefix = bs_prefix


        text_logger.log_id(image_path)

        global_max_score = -1
        global_max_prefix = None
        for step in range(args.tta_steps):
            with torch.no_grad():
                model.eval()
                with torch.cuda.amp.autocast():
                    if step == 0:
                        action = ppo.select_action(bs_prefix.to(device), memory, restart_batch=True)
                        hw_action = hw_ppo.select_action(bs_prefix.to(device), hw_memory, restart_batch=True)

                    else:
                        action = ppo.select_action(bs_prefix.to(device), memory)
                        hw_action = hw_ppo.select_action(bs_prefix.to(device), hw_memory)
                    #
                    patches, patch_size, patch_coordinate = get_patch(ori_image, action, hw_action)  
                    patches = nn.functional.interpolate(patches, size=224, mode='bicubic',
                                                        align_corners=True)


                    reward_model.set_image_features(images=patches)  

                    bs_prefix = clip_model.encode_image(patches).float()  
                    prefix = bs_prefix
                    repeat_prefix = prefix.repeat(sample_k, 1)
                    if args.normalize_prefix: bs_prefix = bs_prefix / bs_prefix.norm(2, -1, keepdim=True)

                    prefix_embed = model.clip_project(bs_prefix).reshape(prefix.shape[0],
                                                                         args.continuous_prompt_length, -1)

                    logits = image_text_simiarlity(texts_embeddings, temperature=args.temperature,
                                                   images_features=global_bs_prefix)
                    detected_objects, _ = top_k_categories(entities_text, logits, args.top_k,
                                                           args.threshold)  

                    detected_objects = detected_objects[0]  

                    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(
                        args.device)

                    discrete_embeddings = model.cap_model.word_embed(discrete_tokens)
                    embeddings = torch.cat((prefix_embed, discrete_embeddings), dim=1)

                    if 'gpt' in args.language_model:

                        sampled_text = beam_search(embeddings=embeddings, tokenizer=tokenizer, beam_width=sample_k,
                                                   model=model.cap_model.gpt)  # List[str]
                    else:

                        sampled_text = opt_generate(model.cap_model.gpt, tokenizer, prompt="", query_embeds=embeddings,
                                                    num_beams=sample_k, device=device, num_captions=sample_k,
                                                    use_nucleus_sampling=False)


                    caption_sampled_text = []
                    sr_sampled_text = []
                    for text in sampled_text:
                        start_index = text.find("[C]") + len("[C]")
                        caption_text = text[start_index:]
                        caption_sampled_text.append(caption_text)
                        sr_end_index = start_index - 4
                        sr_start_index = text.find("[SR]") + len("[SR]")
                        sr_text = text[sr_start_index: sr_end_index]
                        sr_sampled_text.append(sr_text)

                    reward_model.set_text_features(captions=caption_sampled_text)
                    caption_clip_score = reward_model.CLIPScore(
                        text_index=torch.arange(sample_k, dtype=torch.long, device=device), pairwise=False)



                    reward_model.set_text_features(captions=sr_sampled_text)

                    sr_clip_score = reward_model.CLIPScore(
                        text_index=torch.arange(sample_k, dtype=torch.long, device=device), pairwise=False
                    )


                    clip_score = 0.7 * caption_clip_score + 0.3 * sr_clip_score


                    step_max_score = max(clip_score)
                    if step_max_score > global_max_score:
                        global_max_score = step_max_score
                        global_max_prefix = prefix

                    rewards = reward_model.rewards_post_process(
                        clip_score if reward_model.process_batch else clip_score.reshape(1, -1))
                    text_logger.log_sample_text(sampled_text, rewards.tolist())

            model.train()
            optimizer.zero_grad()

            tokenizer.pad_token = tokenizer.eos_token

            return_tokens = tokenizer(sampled_text, return_tensors="pt", padding=True).to(device)

            tokens = return_tokens.input_ids
            atts_opt = torch.ones((len(sampled_text), args.continuous_prompt_length), dtype=torch.long).to(device)
            attention_mask = torch.cat([atts_opt, return_tokens.attention_mask], dim=1)

            with torch.cuda.amp.autocast():
                outputs = model(repeat_prefix, tokens, mask=attention_mask)
                logits = outputs.logits[:, args.continuous_prompt_length - 1: -1]  # 40-end [6, 51, 50272]
                all_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(),
                                           ignore_index=0, reduction='none').reshape(logits.shape[0], -1)
                loss = torch.mean(rewards * all_loss.mean(dim=-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ppo.update(memory)
            hw_ppo.update(hw_memory)

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                prefix_embed = model.clip_project(global_bs_prefix).reshape(global_bs_prefix.shape[0],
                                                                             args.continuous_prompt_length,
                                                                             -1)

                logits = image_text_simiarlity(texts_embeddings, temperature=args.temperature,
                                               images_features=global_bs_prefix)
                detected_objects, _ = top_k_categories(entities_text, logits, args.top_k,
                                                       args.threshold)  # List[List[]], [[category1, category2, ...], [], ...]

                detected_objects = detected_objects[0]  
                discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(
                    args.device)

                discrete_embeddings = model.cap_model.word_embed(discrete_tokens)
                embeddings = torch.cat((prefix_embed, discrete_embeddings), dim=1)

                if 'gpt' in args.language_model:

                    generated_text_prefix = beam_search(embeddings=embeddings, tokenizer=tokenizer, beam_width=sample_k,
                                                        model=model.cap_model.gpt)  # List[str]
                else:
                    generated_text_prefix = opt_generate(model.cap_model.gpt, tokenizer, prompt="",
                                                         query_embeds=embeddings,
                                                         num_beams=5, device=device, num_captions=1)

                sentence = generated_text_prefix[0]
        model.momentum_update_model()
        model.reset_initial()
        optimizer.load_state_dict(optim_state)
        memory.clear_memory()
        hw_memory.clear_memory()

        text_logger.log_final_text(generated_text_prefix)

        start_index = sentence.find("[C]") + len("[C]")
        sentence = sentence[start_index:]


        predict = {}
        predict["split"] = split
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        overall.append(predict)
        if split == 'in_domain':
            indomain.append(predict)
        elif split == 'near_domain':
            neardomain.append(predict)
        elif split == 'out_domain':
            outdomain.append(predict)

    with open(os.path.join(args.out_path, f'overall_generated_captions_{args.suffix}.json'), 'w') as outfile:
        json.dump(overall, outfile, indent=4)
    with open(os.path.join(args.out_path, f'indomain_generated_captions_{args.suffix}.json'), 'w') as outfile:
        json.dump(indomain, outfile, indent=4)
    with open(os.path.join(args.out_path, f'neardomain_generated_captions_{args.suffix}.json'), 'w') as outfile:
        json.dump(neardomain, outfile, indent=4)
    with open(os.path.join(args.out_path, f'outdomain_generated_captions_{args.suffix}.json'), 'w') as outfile:
        json.dump(outdomain, outfile, indent=4)




def tta_coco_flickr30k(args,
                       inpath,
                       entities_text,
                       texts_embeddings,
                       model,
                       tokenizer,
                       clip_model,
                       preprocess,
                       reward_model,
                       optimizer,
                       optim_state,
                       scaler,
                       ppo,
                       hw_ppo,
                       memory,
                       hw_memory,
                       ori_transform,
                       cpu_device
                       ):
    device = args.device

    if args.using_image_features:
        with open(inpath, 'rb') as infile:
            annotations = pickle.load(infile)  
    else:
        with open(inpath, 'r') as infile:
            annotations = json.load(infile) 

    reward_log = os.path.join(args.out_path, f'{args.name_of_datasets}_tta_reward.json')
    text_logger = TxtLogger(reward_log.replace(".json", ".txt"))

    sample_k = reward_model.sample_k

    predicts = []
    for idx, item in enumerate(tqdm(annotations)):
        image_id = item

        captions = annotations[item]

        image_path = args.image_folder + image_id
        image = preprocess(Image.open(image_path)).unsqueeze(dim=0).to(device)  # [1, 3, 224, 224]
        ori_image = ori_transform(Image.open(image_path)).unsqueeze(dim=0).to(device)


        with torch.no_grad():
            model.eval()
            with torch.cuda.amp.autocast():
                bs_prefix = clip_model.encode_image(image).float()  # [1, 512]
                if args.normalize_prefix: bs_prefix = bs_prefix / bs_prefix.norm(2, -1, keepdim=True)
                global_bs_prefix = bs_prefix


        text_logger.log_id(image_path)

        global_max_score = -1
        global_max_prefix = None
        for step in range(args.tta_steps):
            with torch.no_grad():
                model.eval()
                with torch.cuda.amp.autocast():
                    if step == 0:
                        action = ppo.select_action(bs_prefix.to(device), memory, restart_batch=True)
                        hw_action = hw_ppo.select_action(bs_prefix.to(device), hw_memory, restart_batch=True)

                    else:
                        action = ppo.select_action(bs_prefix.to(device), memory)
                        hw_action = hw_ppo.select_action(bs_prefix.to(device), hw_memory)

                    patches, patch_size, patch_coordinate = get_patch(ori_image, action, hw_action)

                    patches = nn.functional.interpolate(patches, size=224, mode='bicubic',
                                                        align_corners=True)

                    reward_model.set_image_features(images=patches)  

                    bs_prefix = clip_model.encode_image(patches).float()  
                    prefix = bs_prefix
                    repeat_prefix = prefix.repeat(sample_k, 1)
                    if args.normalize_prefix: bs_prefix = bs_prefix / bs_prefix.norm(2, -1, keepdim=True)

                    prefix_embed = model.clip_project(bs_prefix).reshape(prefix.shape[0],
                                                                         args.continuous_prompt_length, -1)

                    logits = image_text_simiarlity(texts_embeddings, temperature=args.temperature,
                                                   images_features=global_bs_prefix)
                    detected_objects, _ = top_k_categories(entities_text, logits, args.top_k,
                                                           args.threshold)  

                    detected_objects = detected_objects[
                        0] 

                    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(
                        args.device)

                    discrete_embeddings = model.cap_model.word_embed(discrete_tokens)
                    embeddings = torch.cat((prefix_embed, discrete_embeddings), dim=1)

                    if 'gpt' in args.language_model:

                        sampled_text = beam_search(embeddings=embeddings, tokenizer=tokenizer, beam_width=sample_k,
                                                   model=model.cap_model.gpt)  # List[str]
                    else:

                        sampled_text = opt_generate(model.cap_model.gpt, tokenizer, prompt="", query_embeds=embeddings,
                                                    num_beams=sample_k, device=device, num_captions=sample_k,
                                                    use_nucleus_sampling=False)

                    caption_sampled_text = []
                    sr_sampled_text = []
                    for text in sampled_text:
                        start_index = text.find("[C]") + len("[C]")
                        caption_text = text[start_index:]
                        caption_sampled_text.append(caption_text)
                        sr_end_index = start_index - 4
                        sr_start_index = text.find("[SR]") + len("[SR]")
                        sr_text = text[sr_start_index: sr_end_index]
                        sr_sampled_text.append(sr_text)

                    reward_model.set_text_features(captions=caption_sampled_text)
                    caption_clip_score = reward_model.CLIPScore(
                        text_index=torch.arange(sample_k, dtype=torch.long, device=device), pairwise=False)


                    reward_model.set_text_features(captions=sr_sampled_text)

                    sr_clip_score = reward_model.CLIPScore(
                        text_index=torch.arange(sample_k, dtype=torch.long, device=device), pairwise=False
                    )


                    clip_score = 0.7 * caption_clip_score + 0.3 * sr_clip_score

                    step_max_score = max(clip_score)
                    if step_max_score > global_max_score:
                        global_max_score = step_max_score
                        global_max_prefix = prefix

                    rewards = reward_model.rewards_post_process(
                        clip_score if reward_model.process_batch else clip_score.reshape(1, -1))
                    text_logger.log_sample_text(sampled_text, rewards.tolist())


            model.train()
            optimizer.zero_grad()

            tokenizer.pad_token = tokenizer.eos_token

            return_tokens = tokenizer(sampled_text, return_tensors="pt", padding=True).to(device)

            tokens = return_tokens.input_ids
            atts_opt = torch.ones((len(sampled_text), args.continuous_prompt_length), dtype=torch.long).to(device)
            attention_mask = torch.cat([atts_opt, return_tokens.attention_mask], dim=1)

            with torch.cuda.amp.autocast():
                outputs = model(repeat_prefix, tokens, mask=attention_mask)
                logits = outputs.logits[:, args.continuous_prompt_length - 1: -1]  # 40-end [6, 51, 50272]
                all_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(),
                                           ignore_index=0, reduction='none').reshape(logits.shape[0], -1)
                loss = torch.mean(rewards * all_loss.mean(dim=-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ppo.update(memory)
            hw_ppo.update(hw_memory)

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                prefix_embed = model.clip_project(global_bs_prefix).reshape(global_bs_prefix.shape[0],
                                                                             args.continuous_prompt_length,
                                                                             -1)

                logits = image_text_simiarlity(texts_embeddings, temperature=args.temperature,
                                               images_features=global_bs_prefix)
                detected_objects, _ = top_k_categories(entities_text, logits, args.top_k,
                                                       args.threshold)  # List[List[]], [[category1, category2, ...], [], ...]

                detected_objects = detected_objects[0]  
                discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(
                    args.device)

                discrete_embeddings = model.cap_model.word_embed(discrete_tokens)
                embeddings = torch.cat((prefix_embed, discrete_embeddings), dim=1)

                if 'gpt' in args.language_model:

                    generated_text_prefix = beam_search(embeddings=embeddings, tokenizer=tokenizer, beam_width=sample_k,
                                                        model=model.cap_model.gpt)  # List[str]
                else:
                    generated_text_prefix = opt_generate(model.cap_model.gpt, tokenizer, prompt="",
                                                         query_embeds=embeddings,
                                                         num_beams=5, device=device, num_captions=1)

                sentence = generated_text_prefix[0]
        model.momentum_update_model()
        model.reset_initial()
        optimizer.load_state_dict(optim_state)
        memory.clear_memory()
        hw_memory.clear_memory()

        text_logger.log_final_text(generated_text_prefix)

        start_index = sentence.find("[C]") + len("[C]")
        sentence = sentence[start_index:]


        predict = {}
        predict["split"] = 'valid'
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)

    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions_{args.suffix}.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent=4)


def main(args):
    device = args.device
    cpu_device = torch.device("cpu")
    clip_name = args.clip_model.replace('/', '')
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    if args.name_of_entities_text == 'visual_genome_entities':
        entities_text = load_entities_text(args.name_of_entities_text,
                                           './annotations/vocabulary/all_objects_attributes_relationships.pickle',
                                           not args.disable_all_entities)
        if args.prompt_ensemble:  
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/visual_genome_embedding_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/visual_genome_embedding_{clip_name}.pickle')
    elif args.name_of_entities_text == 'coco_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/coco_categories.json',
                                           not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/coco_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/coco_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'open_image_entities':
        entities_text = load_entities_text(args.name_of_entities_text,
                                           './annotations/vocabulary/oidv7-class-descriptions-boxable.csv',
                                           not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/open_image_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/open_image_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'vinvl_vg_entities':
        entities_text = load_entities_text(args.name_of_entities_text,
                                           './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json',
                                           not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/vg_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/vg_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'vinvl_vgoi_entities':
        entities_text = load_entities_text(args.name_of_entities_text,
                                           './annotations/vocabulary/vgcocooiobjects_v1_class2ind.json',
                                           not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/vgoi_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text,
                                                     f'./annotations/vocabulary/vgoi_embeddings_{clip_name}.pickle')
    else:
        print('The entities text should be input correctly!')
        return

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    cap_model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size,
                                 gpt_type=args.language_model)

    model = CAP_TTA(cap_model, device, momentum_update=args.momentum_update, update_freq=args.update_freq,
                    update_w=args.update_w, momentum=args.tta_momentum, cap_ckpt=args.weight_path)
    model = model.to(device)

    
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    reward_model = get_reward_model(device, args)

    state_dim = 512
    ppo = PPO(feature_dim=512, state_dim=state_dim,
              hidden_state_dim=512, policy_conv=False, device=device)

    hw_ppo = HW_PPO(feature_dim=512, state_dim=state_dim,
                    hidden_state_dim=512, policy_conv=False, device=device)

    memory = Memory()
    hw_memory = HW_Memory()

    optimizer = torch.optim.AdamW(itertools.chain(model.parameters(), ppo.parameters(), hw_ppo.parameters()),
                                  lr=args.tta_lr,
                                  eps=1e-06, weight_decay=args.tta_weight_decay)
    optim_state = deepcopy(optimizer.state_dict())
    ori_image_512transform = img_transform(n_px=512)

    if not args.using_image_features:
        clip_model, preprocess = clip.load(args.clip_model, device=device, download_root='./pretrained/clip')
        inpath = args.path_of_val_datasets
    else:
        inpath = args.path_of_val_datasets[:-5] + f'_{clip_name}.pickle'  # file with image features

    if args.name_of_datasets == 'nocaps':  # nocaps
        tta_nocaps(args, inpath, entities_text, texts_embeddings, model, tokenizer, clip_model, preprocess,
                   reward_model, optimizer, optim_state, scaler, ppo, hw_ppo, memory, hw_memory, ori_image_512transform,
                   None)

    else:  # coco, flickr30k

        tta_coco_flickr30k(args, inpath, entities_text, texts_embeddings, model, tokenizer, clip_model, preprocess,
                           reward_model, optimizer, optim_state, scaler, ppo, hw_ppo, memory, hw_memory,
                           ori_image_512transform, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument('--language_model', default='gpt2')
    parser.add_argument('--continuous_prompt_length', type=int, default=10)
    parser.add_argument('--clip_project_length', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--using_image_features', action='store_true', default=False,
                        help='using pre-extracted image features')
    parser.add_argument('--name_of_datasets', default='coco', choices=('coco', 'flickr30k', 'nocaps'))
    parser.add_argument('--path_of_val_datasets', default='./annotations/coco/val_captions.json')
    parser.add_argument('--disable_all_entities', action='store_true', default=False,
                        help='whether to use entities with a single word only')
    parser.add_argument('--name_of_entities_text', default='vinvl_vgoi_entities', choices=(
        'visual_genome_entities', 'coco_entities', 'open_image_entities', 'vinvl_vg_entities', 'vinvl_vgoi_entities'))
    parser.add_argument('--prompt_ensemble', action='store_true', default=False)
    parser.add_argument('--weight_path', default='./checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_folder', default='./annotations/coco/val2014/')
    parser.add_argument('--out_path', default='./generated_captions.json')
    parser.add_argument('--using_hard_prompt', action='store_true', default=False)
    parser.add_argument('--soft_prompt_first', action='store_true', default=False)
    parser.add_argument('--only_hard_prompt', action='store_true', default=False)
    parser.add_argument('--using_greedy_search', action='store_true', default=False,
                        help='greedy search or beam search')
    parser.add_argument('--noise_mapper', default=None,
                        help='semi noise_mapper during training')
    parser.add_argument('--beam_width', type=int, default=5, help='width of beam')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--text_prompt', type=str, default=None)

    parser.add_argument('--data', default='clip_embedding.pkl',
                        help='path to clip embeddings of captions generated by the attached embeddings_generator script')
    parser.add_argument('--checkpoint', default='./checkpoints/coco_prefix_t10_rn-006.pt')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp",
                        help="Floating point precition.")

    parser.add_argument('--clip_patch', type=int, default=0, help='whether use CLIP patch tokens')

    parser.add_argument('--out_dir', default='./checkpoints', help='path to output directory')
    parser.add_argument('--out_results_file', type=str, default='the output file save the generation file')
    parser.add_argument('--out_clipscore_file', type=str, default='the output file save the generation file')

    parser.add_argument('--add_modality_offset', dest='add_modality_offset', action='store_true', default=False,
                        help='train with modality offset that was pre calculated at others/CLIP_embeddings_centers_info.pkl')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--noise_variance', type=float, default=0.0, help='noise variance')

    parser.add_argument('--uniform_noise', dest='uniform_noise', action='store_true', default=False,
                        help='use uniform noise instead of gaussian')

    parser.add_argument('--batch_size', type=int, default=34, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=5000,
                        help='warm up steps')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1, help='save every n epochs')

    parser.add_argument('--prefix_length', type=int, default=10, help='prefix length')
    parser.add_argument('--prefix_length_clip', type=int, default=10,
                        help='prefix length for clip')

    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true', default=False,
                        help='train only the mapper between CLIP and LLM, while LLM is frozen')
    parser.add_argument('--mapping_type', type=str, default='transformer',
                        help='type of architurctre between CLIP and LLM (mlp/transformer)')
    parser.add_argument('--num_layers', type=int, default=8, help='number of layers in the mapper')
    parser.add_argument('--llm_config_dir', type=str, default=None,
                        help='config and pretrained files of GPT2')

    parser.add_argument('--beam', dest='beam', action='store_true', default=True,
                        help='whether use beam search')

    parser.add_argument('--use_nucleus_sampling', type=int, default=0)
    parser.add_argument('--tta_steps', type=int, default=5, help='number of policy training steps')
    parser.add_argument('--tta_lr', type=float, default=1e-5, help='learning rate of policy gradient')
    parser.add_argument('--tta_weight_decay', default=5e-4, type=float)

    parser.add_argument('--sample_k', type=int, default=5)
    parser.add_argument('--multiple_reward_models', type=int, default=0)
    parser.add_argument('--reward_arch', type=str, default='ViT-L/14')
    parser.add_argument('--reward_process', type=int, default=1,
                        help='If true, process rewards (raw CLIPScore)')
    parser.add_argument('--process_batch', type=int, default=0,
                        help='If true, process rewards through the whole batch (augmentations from a single images)')
    parser.add_argument('--reward_amplify', type=int, default=0)
    parser.add_argument('--weighted_scores', type=int, default=1)

    parser.add_argument('--momentum_update', type=int, default=0,
                        help='If true, update the model in a momentum fashion')
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_w', type=float, default=1.0)
    parser.add_argument('--tta_momentum', type=float, default=0.9999)
    parser.add_argument('--suffix', default='tta_noun_set')

    args = parser.parse_args()

   
    print('\n', vars(args), '\n')

    main(args)
