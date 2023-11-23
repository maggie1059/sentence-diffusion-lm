import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

def load_data(split):
    data_path = 'datasets/wikitext-103/' + split + '.txt'
    # new_file = 'datasets/wikitext-2/wiki-' + split + '.txt'
    new_csv = 'datasets/wikitext-103/wiki-' + split + '.csv'
    new_mask_csv = 'datasets/wikitext-103/wikimask-' + split + '.csv'
    model = SentenceTransformer('all-mpnet-base-v2')
    with open(data_path, encoding="utf-8") as f:
        count = 0
        final = []
        final_mask = []
        for line in f:
            split_text = line.split('|||')
            assert len(split_text) == 2
            assert not split_text[1].isspace()
            sents = split_text[1].split(' . ')

            assert len(sents) == int(split_text[0])
            while len(sents) < 64:
                sents.append(" ")
            input_text = []
            for sent in sents:
                sent = sent + "."
                input_text.append(model.encode(sent))

            mask = np.zeros((64))
            ones = int(split_text[0])
            mask[:ones] = 1

            input_text = np.array(input_text, dtype=np.float64)
            input_text = input_text.flatten()

            if count <= 1:
                print(input_text)
                print(mask)
            count += 1
            
            final.append(input_text)
            final_mask.append(mask)
        final_out = np.stack(final)
        final_mask = np.stack(final_mask)
        # df = pd.DataFrame(final_out)
        # df.to_csv(new_csv)
        np.savetxt(new_csv, final_out, delimiter=",")
        np.savetxt(new_mask_csv, final_mask, delimiter=",")

            # b.write("{}\n".format(input_text.tostring()))


split = 'train'
load_data(split)

new_file = 'datasets/wikitext-103/wiki-' + split + '.csv'
csv = np.genfromtxt(new_file, delimiter=",")
first = csv[0,:]
print(first)
second = csv[1, :]
print(second)

new_mask_file = 'datasets/wikitext-103/wikimask-' + split + '.csv'
csv = np.genfromtxt(new_mask_file, delimiter=",")
first = csv[0,:]
print(first)
second = csv[1, :]
print(second)
# with open(new_file, encoding="utf-8") as f:
#     for line in f:
#         # split_text = line.split('||')
#         # text = split_text[0]
#         # mask = split_text[1]
#         print(np.fromstring(line, dtype=np.float64))
#         # print(np.fromstring(mask, dtype=float))
#         exit()