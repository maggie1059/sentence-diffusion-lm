from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']

# separate sentences by newlines
sentences = []
# with open('./wikitext-2/wiki.valid.tokens', encoding="utf-8") as f:
#     for idx, row in enumerate(f):
#         if row.strip():
#             sentences.append(row)
# sentences = sentences[:64]
MAX_SENTS = 64
new_file = './datasets/wikitext-2/train2.txt'
with open('./datasets/wikitext-2/train.txt', encoding="utf-8") as f, open(new_file, 'w') as b:
    for idx, row in enumerate(f):
        # if idx == 1: break
        if row.strip():
            # print("row: ", row)
            rows = row.split(' . ')[:-1]
            # if len(row) > MAX_SENTS:
            #     continue
            num_ex = len(rows)//MAX_SENTS
            if num_ex*MAX_SENTS < len(rows): num_ex += 1
            for i in range(num_ex):
                # length = min(MAX_SENTS, len(rows))
                num_rows = 0
                if len(rows) < MAX_SENTS:
                    examples = rows
                    num_rows = len(rows)
                else:
                    examples = rows[:MAX_SENTS]
                    num_rows = MAX_SENTS
                # examples = rows[:length]
                    rows = rows[MAX_SENTS:]
                to_write = []
                for sent in examples:
                    # sent = sent + "."
                    to_write.append(sent)
                article = " . ".join(to_write)
                if article[0] == " ":
                    article = article[1:]
                article = article + " ."
                b.write("{}|||{}\n".format(num_rows, article))

            # while len(rows) > 0:
            #     length = min(MAX_SENTS, len(rows))
            #     example = rows[:64]
            #     rows = rows[64:]
            #     for sent in example:
            #         sent = sent + "."
            #     b.write("{}\n".format(article))
            #     row = row[:MAX_SENTS]
            # if len(row) < MAX_SENTS:
            #     row += ['[PAD]' for _ in range(MAX_SENTS-len(row))]
            # row = model.encode(row)
            # print("row shape: ", row.shape)
            # for sent in row:
            #     print("sent: ", sent)
            # sentences.append(row)
            # print("length: ", len(rows))
            # if (len(rows) < 30):
            #     print(rows)

            # print("rows: ", rows)
# print("sentences shape: ", sentences.shape)
exit()
sentences = sentences[:64]

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding.shape)
    print("")

