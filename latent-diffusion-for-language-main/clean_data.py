import os

def generate_examples(data_file):
    new_file = './datasets/wikitext-2/train.txt'
    with open(data_file, encoding="utf-8") as f, open(new_file, 'w') as b:
        # for idx, row in enumerate(f):
        #     if row.strip():
        #         yield idx, {"text": row}
        #     else:
        #         yield idx, {"text": ""}
        # articles = []
        example = []
        for row in f:
            if row.strip():
                # new article
                if '= ' in row and ' =' in row and '= =' not in row:
                    if len(example) > 0:
                        article = "".join(example)
                        article = article.replace("\n", "")
                        # print("article: ", article)
                        
                        b.write("{}\n".format(article))
                    example = []
                elif '= =' not in row:
                    example.append(row)
                # elif header:
                #     throw out
                # else:
                #     split by eos, keep count
                # print("row: ", row)

generate_examples('./datasets/wikitext-2/wiki.train.tokens')