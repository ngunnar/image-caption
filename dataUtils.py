import os
import sys
import numpy as np
import string
import tensorflow as tf
import collections
import random
import math

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def download_file_from_google_drive(id, destination):
    import requests
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_file(filename, source, dest):
    """
    Load file from url
    """
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    print("Downloading %s ... " % filename)
    urlretrieve(source + filename, os.path.join(dest, filename))
    print("done!")


def download_flickr30k(path):
    import tarfile
    if not os.path.exists(os.path.join(path, 'flickr30k_images.tar.gz')):
        download_file_from_google_drive(id='0B_PL6p-5reUAZEM4MmRQQ2VVSlk',
        destination=os.path.join(path, 'flickr30k_images.tar.gz'))
    
    if not os.path.exists(os.path.join(path,'flickr30k_images')):
        with tarfile.open(os.path.join(path,'flickr30k_images.tar.gz'), 'r') as archive:
            archive.extractall(path)

    if not os.path.exists(os.path.join(path, 'flickr30k.tar.gz')):
        download_file('flickr30k.tar.gz', 'http://lixirong.net/data/w2vv-tmm2018/', path)
    
    if not os.path.exists(os.path.join(path, 'flickr30k')):
        with tarfile.open(os.path.join(path, 'flickr30k.tar.gz'), 'r') as archive:
            archive.extractall(path)

    if not os.path.exists(os.path.join(path, 'glove.6B.zip')):
        download_file('glove.6B.zip', 'http://nlp.stanford.edu/data/', path)
        
    if not os.path.exists(os.path.join(path, 'glove.6B')):
        import zipfile
        with zipfile.ZipFile(os.path.join(path, 'glove.6B.zip'), 'r') as zip_ref:
            zip_ref.extractall(path)

def load_flickr30k(path):
    download_flickr30k(path)

    # Train data
    text_file = os.path.join(path, 'flickr30kenctrain', 'TextData', 'flickr30kenctrain.caption.txt')
    glove6B200_file = os.path.join(path, 'glove.6B','glove.6B.200d.txt')
    
    data = collections.defaultdict(list)
    with open(text_file,"r") as f:
        for x in f:
            s = x.split("#enc#")
            sentence = s[1][2:].strip()
            sentence = f"<start> {sentence} <end>"
            data[os.path.join(path, 'flickr30k_images', s[0])].append(sentence)
    
    return data

def convert_to_dataset(data, top_k, split_rate = 0.8):
    train_captions = []
    img_name_vector = []

    for image_path in data:
        caption_list = data[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))
    
    # Convert words to number
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      lower = True,
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    
    # Pad
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = max(len(t) for t in train_seqs) #TODO RETURN?
    
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    img_name_train, cap_train, img_name_val, cap_val = split_dataset(img_name_vector, cap_vector, split_rate)
    
    return img_name_train, cap_train, img_name_val, cap_val, tokenizer, max_length

def split_dataset(img_name_vector, cap_vector, split_rate=0.8):
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(math.ceil(len(img_keys)*split_rate))
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])
    return img_name_train, cap_train, img_name_val, cap_val
