import os
import clip
import pickle
import torch
from tqdm import tqdm


@torch.no_grad()
def main(device: str, clip_type: str, inpath: str, outpath: str):
    device = device
    encoder, _ = clip.load(clip_type, device)

    with open(inpath, 'rb') as infile:
        captions_with_entities = pickle.load(infile) 

    for idx in tqdm(range(len(captions_with_entities))):
        caption = captions_with_entities[idx][1]
        tokens = clip.tokenize(caption, truncate=True).to(device)
        embeddings = encoder.encode_text(tokens).squeeze(dim=0).to('cpu')
        captions_with_entities[idx].append(embeddings)

    with open(outpath, 'wb') as outfile:
        pickle.dump(captions_with_entities, outfile)

    return captions_with_entities


if __name__ == '__main__':

    idx = 0  
    device = 'cuda:0'
    clip_type = 'ViT-B/32'  
    clip_name = clip_type.replace('/', '')

    inpath = [
        './annotations/coco/coco_with_entities_sr.pickle',
        './annotations/flickr30k/flickr30k_with_entities_sr.pickle',
        './annotations/cc3m/cc3m_with_entities.pickle_sr']
    outpath = [
        f'./annotations/coco/coco_texts_sr_features_{clip_name}.pickle',
        f'./annotations/flickr30k/flickr30k_texts_sr_features_{clip_name}.pickle',
        f'./annotations/cc3m/cc3m_texts_sr_features_{clip_name}.pickle']

    if os.path.exists(outpath[idx]):
        with open(outpath[idx], 'rb') as infile:
            captions_with_features = pickle.load(infile)
    else:
        captions_with_features = main(device, clip_type, inpath[idx], outpath[idx])


    print(f'datasets for {inpath[idx]}')
    print(f'The length of datasets: {len(captions_with_features)}')
   
