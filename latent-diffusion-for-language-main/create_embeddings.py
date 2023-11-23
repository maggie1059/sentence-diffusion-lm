MAX_SENTS = 64
new_file = './datasets/wikitext-103/train.txt'
with open('./datasets/wikitext-103/train_full.txt', encoding="utf-8") as f, open(new_file, 'w', encoding="utf-8") as b:
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
