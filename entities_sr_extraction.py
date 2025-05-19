import os
import nltk
import pickle
from typing import List
from nltk.stem import WordNetLemmatizer
from load_annotations import load_captions
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

device = torch.device('cuda:0')


def main(tokenizer, model, captions: List[str], path: str) -> None:
    # writing list file, i.e., [[[entity1, entity2,...], caption], ...]

    model = model.to(device)

    lemmatizer = WordNetLemmatizer()
    new_captions = []
    for caption in tqdm(captions):

        text = tokenizer(
            caption,
            max_length=200,
            return_tensors="pt",
            truncation=True
        )

        text = text.to(device)

        generated_ids = model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id=tokenizer.pad_token_id,
            num_beams=1,
            max_length=200,
            early_stopping=True
        )

        sg_list = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        pattern = r"\((.*?)\)"
        matches = re.findall(pattern, sg_list)

        result = [match.strip() for match in matches]
        filtered_result = [elem for elem in result if not any(char.isdigit() for char in elem)]

        detected_entities = []
        pos_tags = nltk.pos_tag(nltk.word_tokenize(caption))  
        for entities_with_pos in pos_tags:
            if entities_with_pos[1] == 'NN' or entities_with_pos[1] == 'NNS':
                entity = lemmatizer.lemmatize(entities_with_pos[0].lower().strip())
                detected_entities.append(entity)
        detected_entities = list(set(detected_entities))
        new_captions.append([detected_entities, caption, filtered_result])

    with open(path, 'wb') as outfile:
        pickle.dump(new_captions, outfile)


if __name__ == '__main__':

    sg_model = 'pretrained/flan-t5-base-VG-factual-sg-id'

    tokenizer = AutoTokenizer.from_pretrained(sg_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(sg_model)

    datasets = ['coco_captions', 'flickr30k_captions', 'cc3m_train']
    captions_path = [
        './annotations/coco/train_captions.json',
        './annotations/flickr30k/train_captions.json',
        './annotations/cc3m/cc3m_train.json'
    ]
    out_path = [
        './annotations/coco/coco_with_entities_sr.pickle',
        './annotations/flickr30k/flickr30k_with_entities_sr.pickle',
        './annotations/cc3m/cc3m_with_entities_sr.pickle'
    ]

    idx = 2  

    if os.path.exists(out_path[idx]):
        print('Read!')
        with open(out_path[idx], 'rb') as infile:
            captions_with_entities = pickle.load(infile)
        print(f'The length of datasets: {len(captions_with_entities)}')
        captions_with_entities = captions_with_entities[:20]
        for caption_with_entities in captions_with_entities:
            print(caption_with_entities)

    else:
        print('Writing... ...')
        captions = load_captions(datasets[idx], captions_path[idx])
        main(tokenizer, model, captions, out_path[idx])
