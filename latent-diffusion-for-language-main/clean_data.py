import os

def generate_examples(data_file):
    new_file = './datasets/wikitext-103/train_full.txt'
    with open(data_file, encoding="utf-8") as f, open(new_file, 'w', encoding="utf-8") as b:
        example = []
        for row in f:
            if row.strip():
                # new article
                if '= ' in row and ' =' in row and '= =' not in row:
                    if len(example) > 0:
                        article = "".join(example)
                        article = article.replace("\n", "")
                        
                        b.write("{}\n".format(article))
                    example = []
                elif '= =' not in row:
                    example.append(row)

generate_examples('./datasets/wikitext-103/wiki.train.tokens')