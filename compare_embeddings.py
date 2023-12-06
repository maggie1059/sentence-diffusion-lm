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

    print("gt embed shape: ", np.shape(gt_embeds))
    print("gt mask shape: ", np.shape(gt_mask))
    print("gen embed shape: ", np.shape(gen_embeds))
    print("gen mask shape: ", np.shape(gen_mask))
    return gt_embeds, gt_mask, gen_embeds, gen_mask

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
    num_docs = np.shape(gt_embeds)[0]
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

gt_file = 'latent-diffusion-for-language-main/datasets/wikitext-103/wiki-valid.csv'
gt_mask_file = 'latent-diffusion-for-language-main/datasets/wikitext-103/wikimask-valid.csv'
gen_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/2023-12-03_19-21-56/test_sent.csv'
gen_mask_file = 'latent-diffusion-for-language-main/saved_models/wikitext103/2023-12-03_19-21-56/test_mask.csv'

gt_embeds, gt_mask, gen_embeds, gen_mask = load_embeds(gt_file, gt_mask_file, gen_file, gen_mask_file)
# compare_cos(gt_embeds, gt_mask, gen_embeds) # cos similiarity beteween docs + random
# all cos:  [[0.05666438]]
# rand cos:  [[-0.00045218]]
# create_plot(gt_embeds, gt_mask, gen_embeds) # PCA of sentences per doc
# create_seq_plot(gt_embeds, gt_mask, gen_embeds) # PCA colored based on sentence #
# compare_masks(gt_mask, gen_mask) # code to show mask prediction is poor
# compare_cos_seq(gt_embeds, gt_mask, gen_embeds) # cos similarity btwn sequential sentences
# all ground truth cos:  [[0.40616442]]
# all gen cos:  [[0.54360302]]

# TODO: plot pca of all gt vs diffusion to see domains