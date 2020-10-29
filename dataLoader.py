from dataUtils import load_flickr30k, convert_to_dataset

class DataLoader():
    def __init__(self, path, size = 0):
        self.org_data = load_flickr30k(path)
        if size > 0:
            import itertools, collections
            d = collections.defaultdict(list)
            for key, val in itertools.islice(sorted(self.org_data.items(), reverse=True), size):
                d[key] = val
            self.org_data = d

    
    def convert_to_dataset(self, top_k, split_rate = 0.8):
        self.top_k = top_k
        self.img_name_train, self.cap_train, self.img_name_val, self.cap_val, self.tokenizer, self.max_length = convert_to_dataset(self.org_data, top_k, split_rate)