from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

model = SentenceTransformer('all-mpnet-base-v2')

# TODO:
# 1) cosine similarities between gt and generated sentence embeddings
# 2) PCA plot of generated and gt embeddings
# #) PCA plot over time/sequence of generated embeddings (vs gt)


def load_embeds(gt_file, gt_mask_file, gen_file, gen_mask_file):
    gt_embeds = np.genfromtxt(gt_file, delimiter=",")
    gt_embeds = gt_embeds.reshape(-1, 64, 768)

    gt_mask = np.genfromtxt(gt_mask_file, delimiter=",")

    gen_embeds = np.genfromtxt(gen_file, delimiter=",")
    gen_embeds = gen_embeds.reshape(-1, 64, 768)

    gen_mask = np.genfromtxt(gen_mask_file, delimiter=",")

    num_docs = np.shape(gen_embeds)[0]
    gt_embeds = gt_embeds[:num_docs]
    gt_mask = gt_mask[:num_docs]

    print("gt embed shape: ", np.shape(gt_embeds))
    print("gt mask shape: ", np.shape(gt_mask))
    print("gen embed shape: ", np.shape(gen_embeds))
    print("gen mask shape: ", np.shape(gen_mask))
    return gt_embeds, gt_mask, gen_embeds, gen_mask

def load_all_103_embeds(gt_file, gt_mask_file, gen25_file, gen50_file, gen75_file, gen100_file):
    gt_embeds = np.genfromtxt(gt_file, delimiter=",")
    gt_embeds = gt_embeds.reshape(-1, 64, 768)

    gt_mask = np.genfromtxt(gt_mask_file, delimiter=",")

    gen_embeds25 = np.genfromtxt(gen25_file, delimiter=",")
    gen_embeds25 = gen_embeds25.reshape(-1, 64, 768)

    gen_embeds50 = np.genfromtxt(gen50_file, delimiter=",")
    gen_embeds50 = gen_embeds50.reshape(-1, 64, 768)

    gen_embeds75 = np.genfromtxt(gen75_file, delimiter=",")
    gen_embeds75 = gen_embeds75.reshape(-1, 64, 768)

    gen_embeds100 = np.genfromtxt(gen100_file, delimiter=",")
    gen_embeds100 = gen_embeds100.reshape(-1, 64, 768)

    num_docs = np.shape(gen_embeds75)[0]
    gt_embeds = gt_embeds[:num_docs]
    gt_mask = gt_mask[:num_docs]

    print("gt embed shape: ", np.shape(gt_embeds))
    print("gt mask shape: ", np.shape(gt_mask))
    print("gen embed shape: ", np.shape(gen_embeds50))
    return gt_embeds, gt_mask, gen_embeds25, gen_embeds50, gen_embeds75, gen_embeds100

def load_2_103_embeds(gt_file, gt_mask_file, gen_file2, gen_file103):
    gt_embeds = np.genfromtxt(gt_file, delimiter=",")
    gt_embeds = gt_embeds.reshape(-1, 64, 768)

    gt_mask = np.genfromtxt(gt_mask_file, delimiter=",")

    gen_embeds2 = np.genfromtxt(gen_file2, delimiter=",")
    gen_embeds2 = gen_embeds2.reshape(-1, 64, 768)

    gen_embeds103 = np.genfromtxt(gen_file103, delimiter=",")
    gen_embeds103 = gen_embeds103.reshape(-1, 64, 768)

    num_docs = np.shape(gen_embeds103)[0]
    gt_embeds = gt_embeds[:num_docs]
    gt_mask = gt_mask[:num_docs]

    print("gt embed shape: ", np.shape(gt_embeds))
    print("gt mask shape: ", np.shape(gt_mask))
    print("gen embed shape: ", np.shape(gen_embeds2))
    return gt_embeds, gt_mask, gen_embeds2, gen_embeds103

def load_synth_embeds(gt_file, gt_mask_file, synth_file, synth_mask_file):
    gt_embeds = np.genfromtxt(gt_file, delimiter=",")
    gt_embeds = gt_embeds.reshape(-1, 64, 768)

    gt_mask = np.genfromtxt(gt_mask_file, delimiter=",")

    synth_embeds = np.genfromtxt(synth_file, delimiter=",")
    synth_embeds = synth_embeds.reshape(-1, 64, 768)

    synth_mask = np.genfromtxt(synth_mask_file, delimiter=",")

    print("gt embed shape: ", np.shape(gt_embeds))
    print("gt mask shape: ", np.shape(gt_mask))
    print("synth embed shape: ", np.shape(synth_embeds))
    print("synth mask shape: ", np.shape(synth_mask))
    return gt_embeds, gt_mask, synth_embeds, synth_mask

def compare_cos(gt_embeds, gt_mask, gen_embeds):
    num_sents = np.sum(gt_mask, axis=-1)
    print("num sents: ", num_sents)
    num_docs = np.shape(gt_embeds)[0]
    print("num docs: ", num_docs)

    all_cos = 0
    rand_cos = 0

    for doc in range(num_docs):
        print("doc " + str(doc) + " cos:")
        doc_cos = 0
        rand_doc_cos = 0
        num_sent_doc = int(num_sents[doc])
        for sent in range(num_sent_doc):
            ground_truth_sent = gt_embeds[doc, sent].reshape(1, -1)
            rand_sent = np.random.rand(1, 768)
            generated_sent = gen_embeds[doc, sent].reshape(1, -1)

            cos_sim = cosine_similarity(ground_truth_sent, generated_sent)
            doc_cos += cos_sim

            rand_sim = cosine_similarity(ground_truth_sent, rand_sent)
            rand_doc_cos += rand_sim

        doc_cos /= num_sent_doc
        rand_doc_cos /= num_sent_doc
        print(doc_cos)
        print("rand: ", rand_doc_cos)
        all_cos += doc_cos
        rand_cos += rand_doc_cos
    all_cos /= num_docs
    rand_cos /= num_docs
    print("all cos: ", all_cos)
    print("rand cos: ", rand_cos)

def compare_cos_seq(gt_embeds, gt_mask, gen_embeds):
    num_sents = np.sum(gt_mask, axis=-1)
    print("num sents: ", num_sents)
    # num_docs = np.shape(gt_embeds)[0]
    num_docs = np.shape(gen_embeds)[0]
    print("num docs: ", num_docs)

    all_gen_cos = 0
    all_gt_cos = 0
    skipped = 0

    for doc in range(num_docs):
        gen_cos = 0
        gt_cos = 0
        num_sent_doc = int(num_sents[doc])

        for sent in range(num_sent_doc-1):
            gt_sent1 = gt_embeds[doc, sent].reshape(1, -1)
            gt_sent2 = gt_embeds[doc, sent+1].reshape(1, -1)
            gt_sim = cosine_similarity(gt_sent1, gt_sent2)
            gt_cos += gt_sim

            gen_sent1 = gen_embeds[doc, sent].reshape(1, -1)
            gen_sent2 = gen_embeds[doc, sent+1].reshape(1, -1)
            gen_sim = cosine_similarity(gen_sent1, gen_sent2)
            gen_cos += gen_sim

        if num_sent_doc > 1:
            gen_cos /= (num_sent_doc-1)
            gt_cos /= (num_sent_doc-1)
            print("gen " + str(doc) + " cos: ", gen_cos)
            print("ground truth " + str(doc) + " cos: ", gt_cos)

            all_gen_cos += gen_cos
            all_gt_cos += gt_cos
        else:
            skipped += 1

    all_gen_cos /= (num_docs-skipped)
    all_gt_cos /= (num_docs-skipped)
    print("all ground truth cos: ", all_gt_cos)
    print("all gen cos: ", all_gen_cos)

def compare_masks(gt_mask, gen_mask):
    num_docs = np.shape(gt_mask)[0]
    gt_sents = np.sum(gt_mask, axis=-1)
    gen_sents = np.sum(gen_mask, axis=-1)
    print(gen_sents - gt_sents)

    # x = np.arange(num_docs)

    # plt.plot(x, gt_sents)
    # plt.plot(x, gen_sents, '-.')

    # plt.xlabel("Sentence")
    # plt.ylabel("Predicted Length")
    # plt.title("Number of Sentences")
    # plt.show()

def create_plot(gt_embeds, gt_mask, gen_embeds):
    num_docs = np.shape(gt_embeds)[0]
    num_sents_per_doc = np.sum(gt_mask, axis=-1)

    for i in range(num_docs):
        num_sent_doc = int(num_sents_per_doc[i])
        print("num sent doc: ", num_sent_doc)
        gt_embed = gt_embeds[i, :num_sent_doc]
        gen_embed = gen_embeds[i, :num_sent_doc]
        num_sents = np.shape(gt_embed)[0]

        pca = PCA(n_components=2)
        gt_embed = gt_embed.reshape(-1, 768)
        gen_embed = gen_embed.reshape(-1, 768)

        all_data = np.concatenate((gt_embed, gen_embed))
        all_pca = pca.fit_transform(all_data)
        x = all_pca[:,0]
        y = all_pca[:,1]

        colors = ['r'] * num_sents + ['b'] * num_sents

        gt = plt.scatter(x[:num_sents], y[:num_sents], c=colors[:num_sents], alpha=0.2)
        gen = plt.scatter(x[num_sents:], y[num_sents:], c=colors[num_sents:], alpha=0.2)
        plt.legend((gt, gen),
            ('Ground Truth', 'Diffusion'),
            scatterpoints=1,
            loc='lower left')
        plt.title(f'Document {i} Sentence Embeddings')
        plt.savefig(f'pca/embeds_{i}.png', bbox_inches='tight')
        # plt.show()
        plt.clf()

def create_plot_all(gt_embeds, gt_mask, gen_embeds25, gen_embeds50, gen_embeds75, gen_embeds100):
    num_docs = np.shape(gt_embeds)[0]
    num_sents_per_doc = np.sum(gt_mask, axis=-1)

    for i in range(num_docs):
        num_sent_doc = int(num_sents_per_doc[i])
        print("num sent doc: ", num_sent_doc)
        gt_embed = gt_embeds[i, :num_sent_doc]
        gen_embed25 = gen_embeds25[i, :num_sent_doc]
        gen_embed50 = gen_embeds50[i, :num_sent_doc]
        gen_embed75 = gen_embeds75[i, :num_sent_doc]
        gen_embed100 = gen_embeds100[i, :num_sent_doc]
        num_sents = np.shape(gt_embed)[0]

        pca = PCA(n_components=2)
        gt_embed = gt_embed.reshape(-1, 768)
        gen_embed25 = gen_embed25.reshape(-1, 768)
        gen_embed50 = gen_embed50.reshape(-1, 768)
        gen_embed75 = gen_embed75.reshape(-1, 768)
        gen_embed100 = gen_embed100.reshape(-1, 768)

        all_data = np.concatenate((gt_embed, gen_embed25, gen_embed50, gen_embed75, gen_embed100))
        all_pca = pca.fit_transform(all_data)
        x = all_pca[:,0]
        y = all_pca[:,1]

        colors = ['r'] * num_sents + ['b'] * num_sents

        gt = plt.scatter(x[:num_sents], y[:num_sents], c='r', alpha=0.4)
        gen25 = plt.scatter(x[num_sents:num_sents*2], y[num_sents:num_sents*2], c='lightsteelblue', alpha=0.4)
        gen50 = plt.scatter(x[num_sents*2:num_sents*3], y[num_sents*2:num_sents*3], c='cornflowerblue', alpha=0.4)
        gen75 = plt.scatter(x[num_sents*3:num_sents*4], y[num_sents*3:num_sents*4], c='royalblue', alpha=0.4)
        gen100 = plt.scatter(x[num_sents*4:], y[num_sents*4:], c='midnightblue', alpha=0.4)
        plt.legend((gt, gen25, gen50, gen75, gen100),
            ('Ground Truth', 'Diffusion (25k)', 'Diffusion (50k)', 'Diffusion (75k)', 'Diffusion (100k)'),
            scatterpoints=1,
            loc='lower right')
        plt.title(f'Document {i} Sentence Embeddings')
        plt.savefig(f'pca/embeds_{i}.png', bbox_inches='tight')
        # plt.show()
        plt.clf()

def create_plot_datasets(gt_embeds, gt_mask, gen_embeds2, gen_embeds103):
    num_docs = np.shape(gt_embeds)[0]
    num_sents_per_doc = np.sum(gt_mask, axis=-1)

    for i in range(num_docs):
        num_sent_doc = int(num_sents_per_doc[i])
        print("num sent doc: ", num_sent_doc)
        gt_embed = gt_embeds[i, :num_sent_doc]
        gen_embed2 = gen_embeds2[i, :num_sent_doc]
        gen_embed103 = gen_embeds103[i, :num_sent_doc]

        num_sents = np.shape(gt_embed)[0]

        pca = PCA(n_components=2)
        gt_embed = gt_embed.reshape(-1, 768)
        gen_embed2 = gen_embed2.reshape(-1, 768)
        gen_embed103 = gen_embed103.reshape(-1, 768)

        all_data = np.concatenate((gt_embed, gen_embed2, gen_embed103))
        all_pca = pca.fit_transform(all_data)
        x = all_pca[:,0]
        y = all_pca[:,1]

        gt = plt.scatter(x[:num_sents], y[:num_sents], c='r', alpha=0.4)
        gen2 = plt.scatter(x[num_sents:num_sents*2], y[num_sents:num_sents*2], c='g', alpha=0.4)
        gen103 = plt.scatter(x[num_sents*2:], y[num_sents*2:], c='b', alpha=0.4)
        plt.legend((gt, gen2, gen103),
            ('Ground Truth', 'Diffusion (Wikitext-2)', 'Diffusion (Wikitext-103)'),
            scatterpoints=1,
            loc='lower right')
        plt.title(f'Document {i} Sentence Embeddings')
        plt.savefig(f'pca/embeds_{i}.png', bbox_inches='tight')
        # plt.show()
        plt.clf()

def create_plot_datasets_domains(gt_embeds, gt_mask, gen_embeds2, gen_embeds103):
    num_docs = np.shape(gt_embeds)[0]
    num_sents_per_doc = np.sum(gt_mask, axis=-1)
    gt_embeds_list = []
    embeds2 = []
    embeds103 = []

    for i in range(num_docs):
        num_sent_doc = int(num_sents_per_doc[i])
        print("num sent doc: ", num_sent_doc)
        gt_embed = gt_embeds[i, :num_sent_doc]
        gen_embed2 = gen_embeds2[i, :num_sent_doc]
        gen_embed103 = gen_embeds103[i, :num_sent_doc]

        # num_sents = np.shape(gt_embed)[0]

        
        gt_embed = gt_embed.reshape(-1, 768)
        gen_embed2 = gen_embed2.reshape(-1, 768)
        gen_embed103 = gen_embed103.reshape(-1, 768)

        gt_embeds_list.append(gt_embed)
        embeds2.append(gen_embed2)
        embeds103.append(gen_embed103)
    
    embeds2 = np.concatenate(embeds2)
    print("embeds2 shape: ", np.shape(embeds2))
    embeds103 = np.concatenate(embeds103)
    print("embeds103 shape: ", np.shape(embeds103))
    gt_embeds_list = np.concatenate(gt_embeds_list)
    print("gt_embeds_list shape: ", np.shape(gt_embeds_list))

    all_data = np.concatenate((gt_embeds_list, embeds2, embeds103))
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_data)
    x = all_pca[:,0]
    y = all_pca[:,1]
    num_sents = int(len(x)/3)
    print("num sents: ", num_sents)

    gt = plt.scatter(x[:num_sents], y[:num_sents], c='r', alpha=0.1)
    gen2 = plt.scatter(x[num_sents:num_sents*2], y[num_sents:num_sents*2], c='g', alpha=0.1)
    gen103 = plt.scatter(x[num_sents*2:], y[num_sents*2:], c='b', alpha=0.1)
    plt.legend((gt, gen2, gen103),
        ('Ground Truth', 'Diffusion (Wikitext-2)', 'Diffusion (Wikitext-103)'),
        scatterpoints=1,
        loc='upper right')
    plt.title(f'Embedding Domains')
    plt.savefig(f'all_embeds.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

    # all_data = np.concatenate((gt_embeds_list, embeds2))
    # all_pca = pca.fit_transform(all_data)
    # x = all_pca[:,0]
    # y = all_pca[:,1]
    # num_sents = int(len(x)/2)
    # print("num sents: ", num_sents)

    # gt = plt.scatter(x[:num_sents], y[:num_sents], c='r', alpha=0.1)
    # gen2 = plt.scatter(x[num_sents:num_sents*2], y[num_sents:num_sents*2], c='g', alpha=0.1)
    # # gen103 = plt.scatter(x[num_sents:], y[num_sents:], c='b', alpha=0.1)
    # plt.legend((gt, gen2),
    #     ('Ground Truth', 'Diffusion (Wikitext-2)'),
    #     scatterpoints=1,
    #     loc='upper right')
    # plt.title(f'Embedding Domains')
    # plt.savefig(f'all_embeds_all_2.png', bbox_inches='tight')
    # # plt.show()
    # plt.clf()

def create_plot_synth_domains(gt_embeds, gt_mask, synth_embeds, synth_mask):
    num_docs_gt = np.shape(gt_embeds)[0]
    num_docs_synth = np.shape(synth_embeds)[0]
    num_gt_sents_per_doc = np.sum(gt_mask, axis=-1)
    num_synth_sents_per_doc = np.sum(synth_mask, axis=-1)
    gt_embeds_list = []
    synth_embeds_list = []

    for i in range(num_docs_gt):
        num_sent_doc = int(num_gt_sents_per_doc[i])
        gt_embed = gt_embeds[i, :num_sent_doc]

        gt_embed = gt_embed.reshape(-1, 768)

        gt_embeds_list.append(gt_embed)

    for i in range(num_docs_synth):
        num_sent_doc = int(num_synth_sents_per_doc[i])
        synth_embed = synth_embeds[i, :num_sent_doc]

        synth_embed = synth_embed.reshape(-1, 768)

        synth_embeds_list.append(synth_embed)
    
    gt_embeds_list = np.concatenate(gt_embeds_list)
    print("gt_embeds_list shape: ", np.shape(gt_embeds_list))
    synth_embeds_list = np.concatenate(synth_embeds_list)
    print("synth_embeds_list shape: ", np.shape(synth_embeds_list))
    num_synth_sents = np.shape(synth_embeds_list)[0]

    all_data = np.concatenate((gt_embeds_list, synth_embeds_list))
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_data)
    x = all_pca[:,0]
    y = all_pca[:,1]


    gt = plt.scatter(x[:-num_synth_sents], y[:-num_synth_sents], c='r', alpha=0.1)
    gen2 = plt.scatter(x[-num_synth_sents:], y[-num_synth_sents:], c='b', alpha=0.25)
    plt.legend((gt, gen2),
        ('Ground Truth', 'Diffusion (Generated)'),
        scatterpoints=1,
        loc='upper right')
    plt.title(f'Embedding Domains')
    plt.savefig(f'synth_embeds_46.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

def create_seq_plot(gt_embeds, gt_mask, gen_embeds):
    num_docs = np.shape(gt_embeds)[0]
    num_sents_per_doc = np.sum(gt_mask, axis=-1)

    for i in range(num_docs):
        num_sent_doc = int(num_sents_per_doc[i])
        print("num sent doc: ", num_sent_doc)
        gt_embed = gt_embeds[i, :num_sent_doc]
        gen_embed = gen_embeds[i, :num_sent_doc]
        num_sents = np.shape(gt_embed)[0]

        pca = PCA(n_components=2)
        gt_embed = gt_embed.reshape(-1, 768)
        gen_embed = gen_embed.reshape(-1, 768)

        all_data = np.concatenate((gt_embed, gen_embed))
        all_pca = pca.fit_transform(all_data)
        x = all_pca[:,0]
        y = all_pca[:,1]

        colors = list(np.arange(num_sents)) + list(np.arange(num_sents))
        print(colors)

        gt = plt.scatter(x[:num_sents], y[:num_sents], c=colors[:num_sents], alpha=0.6) #cmap='Blues', 
        gen = plt.scatter(x[num_sents:], y[num_sents:], c=colors[num_sents:], alpha=0.6)
        # plt.legend((gt, gen),
        #     ('Ground Truth', 'Diffusion'),
        #     scatterpoints=1,
        #     loc='lower left')
        plt.title(f'Document {i} Sentence Embeddings')
        plt.savefig(f'pca_seq/embeds_{i}.png', bbox_inches='tight')
        # plt.show()
        plt.clf()

def create_synth_seq_plot(synth_embeds, synth_mask):
    num_docs = np.shape(synth_embeds)[0]
    num_sents_per_doc = np.sum(synth_mask, axis=-1)

    for i in range(num_docs):
        num_sent_doc = int(num_sents_per_doc[i])
        print("num sent doc: ", num_sent_doc)
        synth_embed = synth_embeds[i, :num_sent_doc]
        num_sents = np.shape(synth_embed)[0]

        pca = PCA(n_components=2)
        synth_embed = synth_embed.reshape(-1, 768)

        all_pca = pca.fit_transform(synth_embed)
        x = all_pca[:,0]
        y = all_pca[:,1]

        colors = list(np.arange(num_sents))

        gen = plt.scatter(x, y, c=colors, alpha=0.6) #cmap='Blues', 
        plt.title(f'Generated Document {i} Sentence Embeddings')
        plt.savefig(f'synth_pca_seq/embeds_{i}.png', bbox_inches='tight')
        plt.clf()

# gt_file = 'latent-diffusion-for-language-main/datasets/wikitext-103/wiki-valid.csv'
# gt_mask_file = 'latent-diffusion-for-language-main/datasets/wikitext-103/wikimask-valid.csv'
# gen_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/2023-12-03_19-21-56/test_sent.csv'
# gen_mask_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/2023-12-03_19-21-56/test_mask.csv'

# gt_embeds, gt_mask, gen_embeds, gen_mask = load_embeds(gt_file, gt_mask_file, gen_file, gen_mask_file)
# # compare_cos(gt_embeds, gt_mask, gen_embeds) # cos similiarity beteween docs + random
# # all cos:  [[0.05666438]]
# # rand cos:  [[-0.00045218]]
# # create_plot(gt_embeds, gt_mask, gen_embeds) # PCA of sentences per doc
# # create_seq_plot(gt_embeds, gt_mask, gen_embeds) # PCA colored based on sentence #
# # compare_masks(gt_mask, gen_mask) # code to show mask prediction is poor
# # compare_cos_seq(gt_embeds, gt_mask, gen_embeds) # cos similarity btwn sequential sentences
# # all ground truth cos:  [[0.40616442]]
# # all gen cos:  [[0.54360302]]

# TODO: 
# X plot pca of all gt vs diffusion to see domains (for 100k for both datasets)
# X run all above (cos sim, gt vs test gen plots) on each of 6 results
# X plot pca of generated samples vs gt to see domains (maybe with generated test too? to see consistency)
# X seq pca of generated samples for consistency (also seq cos?)

gt_file = 'latent-diffusion-for-language-main/datasets/wikitext-103/wiki-test.csv'
gt_mask_file = 'latent-diffusion-for-language-main/datasets/wikitext-103/wikimask-test.csv'
# gen_file = 'latent-diffusion-for-language-main/saved_models/wikitext/100k_eval/test_sent.csv'
# gen_mask_file = 'latent-diffusion-for-language-main/saved_models/wikitext/100k_eval/test_mask.csv'

synth_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/100k_gen/synth_sent_sample10_seed45.csv'
synth_mask_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/100k_gen/synth_mask_sample10_seed45.csv'

# gt_embeds, gt_mask, gen_embeds, gen_mask = load_embeds(gt_file, gt_mask_file, gen_file, gen_mask_file)

# 103 all PCA
# gen25_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/25k_eval/test_sent.csv'
# gen50_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/50k_eval/test_sent.csv'
# gen75_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/75k_eval/test_sent.csv'
# gen100_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/100k_eval/test_sent.csv'
# gt_embeds, gt_mask, gen_embeds25, gen_embeds50, gen_embeds75, gen_embeds100 = \
#     load_all_103_embeds(gt_file, gt_mask_file, gen25_file, gen50_file, gen75_file, gen100_file)

# create_plot_all(gt_embeds, gt_mask, gen_embeds25, gen_embeds50, gen_embeds75, gen_embeds100)

# datasets PCA
# gen_file2 = 'latent-diffusion-for-language-main/saved_models/wikitext/100k_eval/test_sent.csv'
# gen_file103 = 'latent-diffusion-for-language-main/saved_models/wikitext103/100k_eval/test_sent.csv'

# gt_embeds, gt_mask, gen_embeds2, gen_embeds103 = load_2_103_embeds(gt_file, gt_mask_file, gen_file2, gen_file103)
# create_plot_datasets(gt_embeds, gt_mask, gen_embeds2, gen_embeds103)


# datasets domains PCA
# gen_file2 = 'latent-diffusion-for-language-main/saved_models/wikitext/100k_eval/test_sent.csv'
# gen_file103 = 'latent-diffusion-for-language-main/saved_models/wikitext103/100k_eval/test_sent.csv'

# gt_embeds, gt_mask, gen_embeds2, gen_embeds103 = load_2_103_embeds(gt_file, gt_mask_file, gen_file2, gen_file103)
# create_plot_datasets_domains(gt_embeds, gt_mask, gen_embeds2, gen_embeds103)


# synth PCA
gt_embeds, gt_mask, synth_embeds, synth_mask = load_synth_embeds(gt_file, gt_mask_file, synth_file, synth_mask_file)
# create_plot_synth_domains(gt_embeds, gt_mask, synth_embeds, synth_mask)
create_synth_seq_plot(synth_embeds, synth_mask)


# compare_cos(gt_embeds, gt_mask, gen_embeds) # cos similiarity beteween docs + random
# 25k:
# all cos:  [[0.06757941]]
# rand cos:  [[-6.61753706e-05]]
# 50k:
# all cos:  [[0.06200476]]
# rand cos:  [[-0.00043792]]
# 75k:
# all cos:  [[0.06063424]]
# rand cos:  [[-0.00033598]]
# 100k:
# all cos:  [[0.05831955]]
# rand cos:  [[-0.00016275]]

# wiki2:
# 5k:
# all cos:  [[0.08851514]]
# rand cos:  [[-2.0870338e-05]]
# 100k:
# all cos:  [[0.0888206]]
# rand cos:  [[0.00012406]]

# create_plot(gt_embeds, gt_mask, gen_embeds) # PCA of sentences per doc

# create_seq_plot(gt_embeds, gt_mask, gen_embeds) # PCA colored based on sentence #

# compare_masks(gt_mask, gen_mask) # code to show mask prediction is poor
# [-34. -55. -45. -18.   0.  54.   0.   0. -52. -24. -39.  39.   0.  28.
#   -6.   0. -51.   0.   0.   4. -53.  61. -16. -17. -60. -58.   0. -26.
#  -23.  18. -25.  20. -29. -15.   0.   0.   0. -13.  11.   0.  55.   0.
#   12. -20.   0.   0.  10. -23.   0.   0.  62.   0. -34.   0.   0.  47.
#  -13.  -7.   0.  25. -62.   0. -25.   0. -10.   0.  12.  58.  49. -25.
#    0.   0. -17. -55.   0. -63.   0.   0. -43.   2.  35.  58.  63.  14.
#    0.   0.   8.  17. -62. -50.  26.   5. -51.   0.   0. -19.   0.  10.
#  -56.   0.  -8.  -8.   1.   0.   0.   0.   0.  -8.   0.   0.  -1. -21.
#  -61. -60.   2.   0.   0. -50.   0.   0.   0.  -7. -22.  -5.   0.   0.
#  -19.   0.  36. -62.  40.  18.   0.   0. -52.  54.   0.  48.  30. -12.
#  -50.   0.  -8. -33.  23.  17.   2.   0.]

# compare_cos_seq(gt_embeds, gt_mask, gen_embeds) # cos similarity btwn sequential sentences
# 25k:
# all ground truth cos:  [[0.42560441]]
# all gen cos:  [[0.49280703]]

# 50k:
# all ground truth cos:  [[0.42560441]]
# all gen cos:  [[0.49847176]]

# 75k:
# all ground truth cos:  [[0.42560441]]
# all gen cos:  [[0.53721858]]

# 100k:
# all ground truth cos:  [[0.42560441]]
# all gen cos:  [[0.55872168]]

# wiki2:
# 5k:
# all ground truth cos:  [[0.43355402]]
# all gen cos:  [[0.56274065]]

# 100k:
# all ground truth cos:  [[0.43355402]]
# all gen cos:  [[0.62214637]]
