import os
import json
import pickle
import collections
import re
from nltk.tokenize import word_tokenize
from torchtext.vocab import GloVe, Vectors
from config import Config
import torch
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

pat = re.compile(r"[\w]+|[.,!?;|]")


class MIND_Corpus:
    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'user_ID-%s.json' % config.dataset
        news_ID_file = 'news_ID-%s.json' % config.dataset
        category_file = 'category-%s.json' % config.dataset
        subCategory_file = 'subCategory-%s.json' % config.dataset
        vocabulary_file = 'vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.json'
        word_embedding_file = 'word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl'
        entity_file = 'entity-%s.json' % config.dataset
        entity_embedding_file = 'entity_embedding-%s.pkl' % config.dataset
        context_embedding_file = 'context_embedding-%s.pkl' % config.dataset
        user_history_graph_file = 'user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset + '.pkl'
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file, word_embedding_file, entity_file, entity_embedding_file, context_embedding_file, user_history_graph_file]

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()
            entity_dict = {'<PAD>': 0, '<UNK>': 1}
            news_category_dict = {}

            # 1. user ID dictionay
            with open(os.path.join(config.train_root, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
                for line in train_behaviors_f:
                    impression_ID, user_ID, time, history, impressions = line.split('\t')
                    if user_ID not in user_ID_dict:
                        user_ID_dict[user_ID] = len(user_ID_dict)
                with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                    json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionay & news category dictionay & news subCategory dictionay
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                with open(os.path.join(prefix, 'news_raw.tsv'), 'r', encoding='utf-8') as news_f:
                    for line in news_f:
                        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                        if news_ID not in news_ID_dict:
                            news_ID_dict[news_ID] = len(news_ID_dict)
                            if category not in category_dict:
                                category_dict[category] = len(category_dict)
                            if subCategory not in subCategory_dict:
                                subCategory_dict[subCategory] = len(subCategory_dict)
                            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter: # already appeared in training set
                                            word_counter[word] += 1
                            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter: # already appeared in training set
                                            word_counter[word] += 1
                            for entity in json.loads(title_entities) if title_entities.strip() else []:
                                WikidataId = entity['WikidataId']
                                if WikidataId not in entity_dict:
                                    entity_dict[WikidataId] = len(entity_dict)
                            for entity in json.loads(abstract_entities) if abstract_entities.strip() else []:
                                WikidataId = entity['WikidataId']
                                if WikidataId not in entity_dict:
                                    entity_dict[WikidataId] = len(entity_dict)
                        news_category_dict[news_ID] = category_dict[category]
            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)

            # 3. word dictionay
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True) # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. Word embedding (GloVe for English datasets, Danish FastText for EB-NERD)
            if config.dataset == 'eb-nerd':
                # Danish FastText for EB-NERD dataset
                if config.word_embedding_dim == 300:
                    print("="*60)
                    print("Loading Danish FastTest embedding (cc.da.300.vec)")
                    print("Dataset: EB-NERD (Danish)")
                    print("="*60)
                    glove = Vectors(name='cc.da.300.vec', cache='../../glove_danish')
                else:
                    raise ValueError(f"Danish FastTest only supports 300 dimensions, but got {config.word_embedding_dim}")
            else:
                # GloVe for English datasets(MIND,Adressa,etc.)
                if config.word_embedding_dim == 300:
                    print("="*60)
                    print("Loading GloVe embedding (840B.300d)")
                    print(f"Dataset: {config.dataset} (English)")
                    print("="*60)
                    glove = GloVe(name='840B', dim=300, cache='../../glove', max_vectors=10000000000)
                else:
                    raise ValueError(f"GloVe only supports 300 dimensions, but got{config.word_embedding_dim}")
            
            glove_stoi = glove.stoi
            glove_vectors = glove.vectors
            glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)

            # Scale normalization for Danish FastText
            if config.dataset == 'eb-nerd':
                # GloVe mean norm: 5.89, Danish mean norm: 1.02
                target_norm = 5.89  # GloVe mean
                current_norm = torch.norm(glove_vectors, dim=1).mean().item()
                scaling_factor = target_norm / current_norm

                print(f"\n{'='*60}")
                print(f"Scaling Danish FastText embeddings")
                print(f"  Current mean norm: {current_norm:.4f}")
                print(f"  Target mean norm: {target_norm:.4f}")
                print(f"  Scaling factor: {scaling_factor:.4f}")
                print(f"{'='*60}\n")

                glove_vectors = glove_vectors * scaling_factor
                glove_mean_vector = glove_mean_vector * scaling_factor

            word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
            for word in word_dict:
                index = word_dict[word]
                if index != 0:
                    if word in glove_stoi:
                        word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
                    else:
                        random_vector = torch.zeros(config.word_embedding_dim)
                        random_vector.normal_(mean=0, std=0.1)
                        word_embedding_vectors[index, :] = random_vector + glove_mean_vector
            with open(word_embedding_file, 'wb') as word_embedding_f:
                pickle.dump(word_embedding_vectors, word_embedding_f)

            # 5. knowledge-graph entity dictionary & eneity embedding & context embedding
            entity_embedding_vectors = torch.zeros([len(entity_dict), config.entity_embedding_dim])
            context_embedding_vectors = torch.zeros([len(entity_dict), config.context_embedding_dim])
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                with open(os.path.join(prefix, 'entity_embedding.vec'), 'r', encoding='utf-8') as entity_f:
                    for line in entity_f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split('\t')
                            assert len(terms) == config.entity_embedding_dim + 1, 'entity embedding dim does not match'
                            WikidataId = terms[0]
                            if WikidataId in entity_dict:
                                entity_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                with open(os.path.join(prefix, 'context_embedding.vec'), 'r', encoding='utf-8') as context_f:
                    for line in context_f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split('\t')
                            assert len(terms) == config.context_embedding_dim + 1, 'context embedding dim does not match'
                            WikidataId = terms[0]
                            if WikidataId in entity_dict:
                                context_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
            with open(entity_file, 'w', encoding='utf-8') as entity_f:
                json.dump(entity_dict, entity_f)
            with open(entity_embedding_file, 'wb') as entity_embedding_f:
                pickle.dump(entity_embedding_vectors, entity_embedding_f)
            with open(context_embedding_file, 'wb') as context_embedding_f:
                pickle.dump(context_embedding_vectors, context_embedding_f)

            # 6. user history graph
            category_num = len(category_dict)
            graph_size = config.max_history_num + category_num # graph size of |V_{n}|+|V_{p}|
            prefix_mode = ['train', 'dev', 'test']
            user_history_graph_data = {}
            for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                mode = prefix_mode[prefix_index]
                user_history_num = 0
                with open(os.path.join(prefix, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as behaviors_f:
                    for line in behaviors_f:
                        user_history_num += 1
                user_history_graph = np.zeros([user_history_num, graph_size, graph_size], dtype=np.float32)
                user_history_category_mask = np.zeros([user_history_num, category_num + 1], dtype=bool)
                user_history_category_indices = np.zeros([user_history_num, config.max_history_num], dtype=np.int64)
                with open(os.path.join(prefix, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as behaviors_f:
                    for line_index, line in enumerate(behaviors_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        if config.no_self_connection:
                            history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
                        else:
                            history_graph = np.identity(graph_size, dtype=np.float32)
                        history_category_mask = np.zeros(category_num + 1, dtype=bool) # extra one category index for padding news
                        history_category_indices = np.full([config.max_history_num], category_num, dtype=np.int64)
                        if len(history.strip()) > 0:
                            history_news_ID = history.split(' ')
                            offset = max(0, len(history_news_ID) - config.max_history_num)
                            history_news_num = min(len(history_news_ID), config.max_history_num)
                            for i in range(history_news_num):
                                category_index = news_category_dict[history_news_ID[i + offset]]
                                history_category_mask[category_index] = 1
                                history_category_indices[i] = category_index
                                history_graph[i, config.max_history_num + category_index] = 1 # edge of E_{p}^{1} in inter-cluster graph G2
                                history_graph[config.max_history_num + category_index, i] = 1 # edge of E_{p}^{1} in inter-cluster graph G2
                                for j in range(i + 1, history_news_num):
                                    _category_index = news_category_dict[history_news_ID[j + offset]]
                                    if category_index == _category_index:
                                        history_graph[i, j] = 1 # edge of E_{n} in intra-cluster graph G1
                                        history_graph[j, i] = 1 # edge of E_{n} in intra-cluster graph G1
                                    else:
                                        history_graph[config.max_history_num + category_index, config.max_history_num + _category_index] = 1 # edge of E_{p}^{2} in inter-cluster graph G2
                                        history_graph[config.max_history_num + _category_index, config.max_history_num + category_index] = 1 # edge of E_{p}^{2} in inter-cluster graph G2
                            if not config.no_adjacent_normalization:
                                if config.gcn_normalization_type == 'asymmetric':
                                    # Asymmetric adjacent matrix normalization: D^{-\frac{1}{2}}A
                                    D_inv = np.zeros([graph_size, graph_size], dtype=np.float32)
                                    np.fill_diagonal(D_inv, 1 / history_graph.sum(axis=1, keepdims=False))
                                    history_graph = np.matmul(D_inv, history_graph)
                                else:
                                    # Symmetric adjacent matrix normalization: D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
                                    D_inv_sqrt = np.zeros([graph_size, graph_size], dtype=np.float32)
                                    np.fill_diagonal(D_inv_sqrt, np.sqrt(1 / history_graph.sum(axis=1, keepdims=False)))
                                    history_graph = np.matmul(np.matmul(D_inv_sqrt, history_graph), D_inv_sqrt)
                        user_history_graph[line_index] = history_graph
                        user_history_category_mask[line_index] = history_category_mask
                        user_history_category_indices[line_index] = history_category_indices
                    user_history_graph_data[mode + '_user_history_graph'] = user_history_graph
                    user_history_graph_data[mode + '_user_history_category_mask'] = user_history_category_mask
                    user_history_graph_data[mode + '_user_history_category_indices'] = user_history_category_indices
            with open(user_history_graph_file, 'wb') as user_history_graph_f:
                pickle.dump(user_history_graph_data, user_history_graph_f)

    def __init__(self, config: Config):
        # preprocess data
        MIND_Corpus.preprocess(config)
        with open('user_ID-%s.json' % config.dataset, 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)
        with open('news_ID-%s.json' % config.dataset, 'r', encoding='utf-8') as news_ID_f:
            self.news_ID_dict = json.load(news_ID_f)
            self.news_num = len(self.news_ID_dict)
        with open('category-%s.json' % config.dataset, 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open('subCategory-%s.json' % config.dataset, 'r', encoding='utf-8') as subCategory_f:
            self.subCategory_dict = json.load(subCategory_f)
            config.subCategory_num = len(self.subCategory_dict)
        with open('vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.json', 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)
        with open('entity-%s.json' % config.dataset, 'r', encoding='utf-8') as entity_f:
            self.entity_dict = json.load(entity_f)
            config.entity_size = len(self.entity_dict)
        with open('user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset + '.pkl', 'rb') as user_history_graph_f:
            user_history_data = pickle.load(user_history_graph_f)
            self.train_user_history_graph = user_history_data['train_user_history_graph']
            self.train_user_history_category_mask = user_history_data['train_user_history_category_mask']
            self.train_user_history_category_indices = user_history_data['train_user_history_category_indices']
            self.dev_user_history_graph = user_history_data['dev_user_history_graph']
            self.dev_user_history_category_mask = user_history_data['dev_user_history_category_mask']
            self.dev_user_history_category_indices = user_history_data['dev_user_history_category_indices']
            self.test_user_history_graph = user_history_data['test_user_history_graph']
            self.test_user_history_category_mask = user_history_data['test_user_history_category_mask']
            self.test_user_history_category_indices = user_history_data['test_user_history_category_indices']

        # meta data
        self.negative_sample_num = config.negative_sample_num                                           # negative sample number for training
        self.max_history_num = config.max_history_num                                                   # max history number for each training user
        self.max_title_length = config.max_title_length                                                 # max title length for each news text
        self.max_abstract_length = config.max_abstract_length                                           # max abstract length for each news text
        self.news_category = np.zeros([self.news_num], dtype=np.int32)                                  # [news_num]
        self.news_subCategory = np.zeros([self.news_num], dtype=np.int32)                               # [news_num]
        self.news_title_text = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)         # [news_num, max_title_length]
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length], dtype=bool)             # [news_num, max_title_length]
        self.news_title_entity = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)       # [news_num, max_title_length]
        self.news_abstract_text = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32)   # [news_num, max_abstract_length]
        self.news_abstract_mask = np.zeros([self.news_num, self.max_abstract_length], dtype=bool)       # [news_num, max_abstract_length]
        self.news_abstract_entity = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32) # [news_num, max_abstract_length]
        self.news_popularity = np.zeros([self.news_num], dtype=np.float32)                             # [news_num] - PENR popularity labels
        self.news_popularity_class = np.zeros([self.news_num], dtype=np.int32)                     # [news_num] - POPCORN popularity class labels (0..popcorn_num_pop_classes-1)
        self.train_behaviors = []                                                                       # [user_ID, [history], [history_mask], click impression, [non-click impressions], behavior_index]
        self.dev_behaviors = []                                                                         # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.dev_indices = []                                                                           # index for dev
        self.test_behaviors = []                                                                        # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.test_indices = []                                                                          # index for test
        self.title_word_num = 0
        self.abstract_word_num = 0
        
        # PLM-NR
        self.news_title_texts = [''] * self.news_num        # added: stores raw title text (for PLM)
        self.news_abstract_texts = [''] * self.news_num     # added: stores raw abstract text (for PLM)

        # generate news meta data
        news_ID_set = set(['<PAD>'])
        news_lines = []
        with open(os.path.join(config.train_root, 'news_raw.tsv'), 'r', encoding='utf-8') as train_news_f:
            for line in train_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        with open(os.path.join(config.dev_root, 'news_raw.tsv'), 'r', encoding='utf-8') as dev_news_f:
            for line in dev_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        with open(os.path.join(config.test_root, 'news_raw.tsv'), 'r', encoding='utf-8') as test_news_f:
            for line in test_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        assert self.news_num == len(news_ID_set), 'news num mismatch %d v.s. %d' % (self.news_num, len(news_ID_set))
        for line in news_lines:
            news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
            index = self.news_ID_dict[news_ID]
            self.news_category[index] = self.category_dict[category] if category in self.category_dict else 0
            self.news_subCategory[index] = self.subCategory_dict[subCategory] if subCategory in self.subCategory_dict else 0

            # PLM-NR
            self.news_title_texts[index] = title  # added: store raw title text
            self.news_abstract_texts[index] = abstract  # added: store raw abstract text

            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
            offsets = [-1 for _ in range(len(title))]
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_title_length:
                    break
                if is_number(word):
                    self.news_title_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_title_text[index][i] = self.word_dict[word]
                else:
                    self.news_title_text[index][i] = 1
                self.news_title_mask[index][i] = 1
                while title[offset_index] in [' ', '\t']:
                    offset_index += 1
                for j in range(len(word)):
                    offsets[offset_index] = i
                    offset_index += 1
            for entity in json.loads(title_entities) if title_entities.strip() else []:
                WikidataId = entity['WikidataId']
                for offset in entity['OccurrenceOffsets']:
                    if 0 <= offset < len(offsets) and offsets[offset] != -1 and WikidataId in self.entity_dict:
                        self.news_title_entity[index][offsets[offset]] = self.entity_dict[WikidataId]
            self.title_word_num += len(words)
            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
            offsets = [-1 for _ in range(len(abstract))]
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_abstract_length:
                    break
                if is_number(word):
                    self.news_abstract_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_abstract_text[index][i] = self.word_dict[word]
                else:
                    self.news_abstract_text[index][i] = 1
                self.news_abstract_mask[index][i] = 1
                while abstract[offset_index] in [' ', '\t']:
                    offset_index += 1
                for j in range(len(word)):
                    offsets[offset_index] = i
                    offset_index += 1
            for entity in json.loads(abstract_entities) if abstract_entities.strip() else []:
                WikidataId = entity['WikidataId']
                for offset in entity['OccurrenceOffsets']:
                    if 0 <= offset < len(offsets) and offsets[offset] != -1 and WikidataId in self.entity_dict:
                        self.news_abstract_entity[index][offsets[offset]] = self.entity_dict[WikidataId]
            self.abstract_word_num += len(words)
        self.news_title_mask[0][0] = 1    # for <PAD> news
        self.news_abstract_mask[0][0] = 1 # for <PAD> news

        # generate behavior meta data
        with open(os.path.join(config.train_root, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
            for behavior_index, line in enumerate(train_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                click_impressions = []
                non_click_impressions = []
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        click_impressions.append(self.news_ID_dict[impression[:-2]])
                    else:
                        non_click_impressions.append(self.news_ID_dict[impression[:-2]])
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for click_impression in click_impressions:
                        self.train_behaviors.append([self.user_ID_dict[user_ID], user_history, user_history_mask, click_impression, non_click_impressions, behavior_index])
                else:
                    for click_impression in click_impressions:
                        self.train_behaviors.append([self.user_ID_dict[user_ID], [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), click_impression, non_click_impressions, behavior_index])
        with open(os.path.join(config.dev_root, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as dev_behaviors_f:
            for dev_ID, line in enumerate(dev_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression[:-2]], dev_ID])
                else:
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression[:-2]], dev_ID])
        with open(os.path.join(config.test_root, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as test_behaviors_f:
            for test_ID, line in enumerate(test_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        if config.dataset != 'large':
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression[:-2]], test_ID])
                        else:
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression], test_ID])
                else:
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        if config.dataset != 'large':
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression[:-2]], test_ID])
                        else:
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression], test_ID])

        # Compute news popularity for PENR (Equation 16 labels)
        if config.click_predictor == 'PENR':
            self._compute_news_popularity(config)

        # Compute news popularity class labels for POPCORN
        # NOTE: POPCORN also needs _compute_news_popularity() first to compute self.news_popularity
        if config.click_predictor == 'POPCORN':
            self._compute_news_popularity(config)  # First compute popularity values and click_counts
            class_method = getattr(config, 'popcorn_class_method', 'fixed')
            self._compute_news_popularity_class(config, class_method=class_method)  # Then compute class labels

        # Compute TCCM statistics (publish_time, entity/word CTR tables)
        # Activated when click_predictor == 'TCCM' OR (click_predictor == 'POPCORN' and use_tccm_addon)
        # → no impact on other models
        if config.click_predictor == 'TCCM' or \
           (config.click_predictor == 'POPCORN' and getattr(config, 'use_tccm_addon', False)):
            self._compute_tccm_statistics(config)

        if config.use_plm_news_encoder:
            self._preprocess_for_plm(config)

    def _compute_news_popularity(self, config=None):
        """
        Compute news popularity labels for PENR/POPCORN.
        Popularity is computed as normalized click count from training behaviors.

        norm_scope='topic' (default):
          - Each news normalized by its topic's max count → [0.0, 1.0]
          - Stores result in self.news_popularity
        norm_scope='global':
          - No normalization applied; self.news_popularity stays at -1.0
          - quantile_rank method uses self.news_click_counts (raw counts) directly,
            so news_popularity is not needed in this mode

        IMPORTANT: Only clicked news are counted. News with 0 clicks are set to -1.0.

        Output:
        - self.news_click_counts: {news_id: raw_count} — always populated
        - self.news_popularity: [news_num] float array
            - topic mode: [0.0, 1.0] for clicked news, -1.0 for unclicked
            - global mode: -1.0 for all (unused by quantile_rank)
        """
        norm_scope = getattr(config, 'popcorn_pop_normalization', 'topic') if config is not None else 'topic'
        print(f'Computing news popularity labels (norm_scope={norm_scope})...')

        # Use dictionary to count only clicked news
        click_counts = {}  # {news_id: count}

        # Track history per user to avoid duplicate counting
        user_history_set = {}  # {user_id: set of news_ids}

        # Count clicks from training behaviors
        for behavior in self.train_behaviors:
            user_id = behavior[0]
            clicked_news = behavior[3]  # Positive news index
            history_indices = behavior[1]  # [max_history_num]

            # Count clicked impression (each impression click counted)
            if clicked_news not in click_counts:
                click_counts[clicked_news] = 0
            click_counts[clicked_news] += 1

            # Collect history per user (will deduplicate later)
            if user_id not in user_history_set:
                user_history_set[user_id] = set()

            for idx in history_indices:
                if idx > 0:  # Skip padding (index 0 is <PAD>)
                    user_history_set[user_id].add(idx)

        # Count history clicks (deduplicated per user)
        for user_id, history_set in user_history_set.items():
            for news_id in history_set:
                if news_id not in click_counts:
                    click_counts[news_id] = 0
                click_counts[news_id] += 1

        # Store raw click counts (used by quantile_rank in both norm modes)
        self.news_click_counts = click_counts.copy()

        # Initialize with -1.0 (unclicked news will remain -1.0)
        self.news_popularity = np.full(self.news_num, -1.0, dtype=np.float32)

        # Always build topic grouping (needed for stats and topic normalization)
        topic_click_counts = {}   # {topic_id: {news_id: count}}
        topic_max_counts = {}     # {topic_id: max_count}

        if click_counts:
            for news_id, count in click_counts.items():
                if news_id > 0:  # Skip <PAD> (index 0)
                    topic = self.news_category[news_id]
                    if topic not in topic_click_counts:
                        topic_click_counts[topic] = {}
                    topic_click_counts[topic][news_id] = count

            # Always compute topic max counts (used for stats in both modes)
            for topic, news_counts in topic_click_counts.items():
                if news_counts:
                    topic_max_counts[topic] = max(news_counts.values())

            if norm_scope == 'topic':
                # Normalize each news by its topic's max count → [0.0, 1.0]
                for topic, news_counts in topic_click_counts.items():
                    topic_max = topic_max_counts[topic]
                    if topic_max > 0:
                        for news_id, count in news_counts.items():
                            self.news_popularity[news_id] = count / topic_max
            # global: news_popularity stays at -1.0 (quantile_rank uses news_click_counts directly)

        clicked_count = len(click_counts)
        zero_click_count = self.news_num - clicked_count

        # Compute statistics
        topic_stats = {}
        for topic in topic_click_counts.keys():
            topic_news_ids = list(topic_click_counts[topic].keys())
            if norm_scope == 'topic':
                topic_vals = [self.news_popularity[nid] for nid in topic_news_ids]
            else:
                topic_vals = [click_counts[nid] for nid in topic_news_ids]
            topic_stats[topic] = {
                'count': len(topic_news_ids),
                'max_count': topic_max_counts.get(topic, 0),
                'mean_val': np.mean(topic_vals) if topic_vals else 0.0
            }

        print(f'Popularity computed (norm_scope={norm_scope}):')
        print(f'  - Total news: {self.news_num}')
        print(f'  - Clicked news: {clicked_count}')
        print(f'  - Zero-click news (excluded): {zero_click_count}')
        print(f'  - Number of topics: {len(topic_click_counts)}')
        if norm_scope == 'topic':
            clicked_pops = [self.news_popularity[nid] for nid in click_counts.keys()]
            print(f'  - Mean popularity (clicked only): {np.mean(clicked_pops):.4f}')
        else:
            global_max = max(click_counts.values()) if click_counts else 0
            print(f'  - Global max click count: {global_max}')
            print(f'  - Mean click count (clicked only): {np.mean(list(click_counts.values())):.2f}')
        print(f'  - Unique users with history: {len(user_history_set)}')
        if len(topic_stats) <= 10:
            print(f'  - Topic statistics:')
            for topic, stats in sorted(topic_stats.items()):
                print(f'    Topic {topic}: {stats["count"]} news, max_count={stats["max_count"]}, mean_val={stats["mean_val"]:.4f}')

    def _compute_news_popularity_class(self, config: Config, class_method='fixed'):
        """
        Compute news popularity class labels for POPCORN model.
        Supports two methods: 'fixed' (original) and 'quantile_rank' (new).

        Method 1: 'fixed' (original)
        - Uses pre-computed self.news_popularity (normalized by topic max)
        - Fixed-range binning: divides [0.0, 1.0] into num_classes equal intervals
        - Class assignment based on absolute popularity value

        Method 2: 'quantile_rank' (improved)
        - Uses raw click counts (no normalization)
        - Groups news by topic, sorts by click count within each topic
        - Computes average rank for ties (same count → same rank)
        - Converts rank to percentile, then maps to class
        - Handles small topics (1 news → middle class)

        Process for 'fixed':
        1. Use pre-computed self.news_popularity (already normalized by topic max)
        2. Divide popularity range [0.0, 1.0] into num_classes equal intervals
        3. Assign class based on which interval the popularity value falls into

        Process for 'quantile_rank':
        1. Use raw click counts from self.news_click_counts (no normalization)
        2. Group clicked news by topic category
        3. Sort news by click count within each topic (ascending)
        4. Compute average rank for ties (same count → same avg rank)
        5. percentile = avg_rank / (num_news - 1)  →  [0.0, 1.0]
        6. class = min(int(percentile * num_classes), num_classes - 1)
        7. Small topic (1 news) → assign middle class (num_classes // 2)

        IMPORTANT: 
        - Only clicked news get class labels (0~num_classes-1). Unclicked news are -1.
        - 'fixed' method preserves absolute popularity meaning
        - 'quantile_rank' method ensures: same count → same class (tie consistency)

        Output:
        - self.news_popularity_class: [news_num] array of class labels
          - Clicked news: 0 to num_classes-1
          - Unclicked news: -1 (ignored in loss calculation)
        """
        num_classes = config.popcorn_num_pop_classes

        # Initialize with -1 (unclicked news will remain -1)
        self.news_popularity_class = np.full(self.news_num, -1, dtype=np.int64)

        if class_method == 'fixed':
            # Original method: fixed-range binning using normalized popularity
            print(f'Computing news popularity class labels for POPCORN (fixed-range binning, using topic-aligned popularity)...')
            
            # Compute interval size: [0.0, 1.0] divided into num_classes
            interval_size = 1.0 / num_classes

            # Assign classes based on fixed popularity ranges
            for news_idx in range(1, self.news_num):  # Skip 0 (<PAD>)
                # Only process clicked news (popularity >= 0)
                if self.news_popularity[news_idx] >= 0:
                    pop_value = self.news_popularity[news_idx]
                    
                    # Calculate which interval the popularity falls into
                    # Handle edge case: pop_value == 1.0 should be in the last class
                    if pop_value >= 1.0:
                        class_label = num_classes - 1
                    else:
                        class_label = int(pop_value / interval_size)
                        # Ensure class_label is in valid range [0, num_classes-1]
                        class_label = min(class_label, num_classes - 1)
                    
                    self.news_popularity_class[news_idx] = class_label

            # <PAD> news (index 0)
            self.news_popularity_class[0] = -1

            valid_count = np.sum(self.news_popularity_class >= 0)
            ignored_count = np.sum(self.news_popularity_class == -1)

            # Compute class distribution and statistics
            valid_classes = self.news_popularity_class[self.news_popularity_class >= 0]
            class_distribution = np.bincount(valid_classes, minlength=num_classes) if len(valid_classes) > 0 else np.zeros(num_classes, dtype=np.int64)
            
            # Compute popularity range for each class
            class_ranges = []
            for i in range(num_classes):
                lower = i * interval_size
                upper = (i + 1) * interval_size if i < num_classes - 1 else 1.0
                class_ranges.append((lower, upper))

            # Compute and print popularity distribution
            valid_popularities = self.news_popularity[self.news_popularity >= 0]
            if len(valid_popularities) > 0:
                # Create histogram with num_classes bins
                hist, bin_edges = np.histogram(valid_popularities, bins=num_classes, range=(0.0, 1.0))
                
                print(f'  - Popularity distribution (clicked news only):')
                print(f'    Min: {np.min(valid_popularities):.4f}, Max: {np.max(valid_popularities):.4f}')
                print(f'    Mean: {np.mean(valid_popularities):.4f}, Median: {np.median(valid_popularities):.4f}')
                print(f'    Std: {np.std(valid_popularities):.4f}')
                print(f'    Popularity histogram (bins={num_classes}):')
                for i in range(num_classes):
                    bin_lower = bin_edges[i]
                    bin_upper = bin_edges[i + 1]
                    count = hist[i]
                    percentage = (count / len(valid_popularities)) * 100 if len(valid_popularities) > 0 else 0
                    bar = '█' * int(count / max(hist) * 20) if max(hist) > 0 else ''  # Visual bar (max 20 chars)
                    print(f'      [{bin_lower:.3f}, {bin_upper:.3f}): {count:6d} ({percentage:5.2f}%) {bar}')

            print(f'  - Popularity classes computed: {num_classes} classes (fixed-range binning)')
            print(f'  - Interval size: {interval_size:.4f}')
            print(f'  - Valid news: {valid_count}')
            print(f'  - Ignored news (0 clicks + padding): {ignored_count}')
            print(f'  - Class distribution (valid only): {class_distribution}')
            if num_classes <= 10:  # Print ranges if not too many classes
                print(f'  - Class ranges:')
                for i, (lower, upper) in enumerate(class_ranges):
                    count = class_distribution[i] if i < len(class_distribution) else 0
                    print(f'    Class {i}: [{lower:.3f}, {upper:.3f}] -> {count} news')

        elif class_method == 'quantile_rank':
            if not hasattr(self, 'news_click_counts') or self.news_click_counts is None:
                raise ValueError("news_click_counts not found. Make sure _compute_news_popularity() is called first.")

            click_counts = self.news_click_counts
            norm_scope = getattr(config, 'popcorn_pop_normalization', 'topic')
            tie_stats = {'total_ties': 0, 'total_tie_groups': 0}

            def _assign_quantile_ranks(news_list_sorted):
                """sorted [(news_id, count)] → assign class labels via avg-rank percentile."""
                num_news = len(news_list_sorted)
                if num_news == 1:
                    self.news_popularity_class[news_list_sorted[0][0]] = num_classes // 2
                    return
                avg_ranks = [0.0] * num_news
                i = 0
                while i < num_news:
                    j = i
                    while j < num_news and news_list_sorted[j][1] == news_list_sorted[i][1]:
                        j += 1
                    avg_rank = (i + j - 1) / 2.0
                    if j - i > 1:
                        tie_stats['total_tie_groups'] += 1
                        tie_stats['total_ties'] += (j - i)
                    for k in range(i, j):
                        avg_ranks[k] = avg_rank
                    i = j
                for idx in range(num_news):
                    news_id = news_list_sorted[idx][0]
                    percentile = avg_ranks[idx] / (num_news - 1)
                    self.news_popularity_class[news_id] = min(int(percentile * num_classes), num_classes - 1)

            if norm_scope == 'global':
                print(f'Computing news popularity class labels for POPCORN (global quantile_rank)...')
                # All clicked news ranked together regardless of topic
                all_news_sorted = sorted(
                    [(nid, cnt) for nid, cnt in click_counts.items() if nid > 0],
                    key=lambda x: x[1]
                )
                _assign_quantile_ranks(all_news_sorted)
                topic_news = {}  # still build for stats
                for news_id, count in click_counts.items():
                    if news_id > 0:
                        topic = self.news_category[news_id]
                        if topic not in topic_news:
                            topic_news[topic] = []
                        topic_news[topic].append((news_id, count))

            else:  # norm_scope == 'topic'
                print(f'Computing news popularity class labels for POPCORN (topic quantile_rank)...')
                # Group clicked news by topic, rank within each topic
                topic_news = {}
                for news_id, count in click_counts.items():
                    if news_id > 0:
                        topic = self.news_category[news_id]
                        if topic not in topic_news:
                            topic_news[topic] = []
                        topic_news[topic].append((news_id, count))

                for topic, news_list in topic_news.items():
                    if len(news_list) == 0:
                        continue
                    news_list_sorted = sorted(news_list, key=lambda x: x[1])
                    _assign_quantile_ranks(news_list_sorted)

            # <PAD> news (index 0)
            self.news_popularity_class[0] = -1

            valid_count = np.sum(self.news_popularity_class >= 0)
            ignored_count = np.sum(self.news_popularity_class == -1)

            valid_classes = self.news_popularity_class[self.news_popularity_class >= 0]
            class_distribution = np.bincount(valid_classes, minlength=num_classes) if len(valid_classes) > 0 else np.zeros(num_classes, dtype=np.int64)

            # Click count range per class (for stats)
            class_click_ranges = []
            for class_idx in range(num_classes):
                class_mask = (self.news_popularity_class >= 0) & (self.news_popularity_class == class_idx)
                class_news_ids = np.where(class_mask)[0]
                if len(class_news_ids) > 0:
                    class_counts = [click_counts.get(int(nid), 0) for nid in class_news_ids]
                    class_click_ranges.append((min(class_counts), max(class_counts)))
                else:
                    class_click_ranges.append((0, 0))

            # Topic-wise class distribution (for stats)
            topic_class_stats = {}
            for topic, news_list in topic_news.items():
                topic_news_ids = [nid for nid, _ in news_list]
                topic_classes = [self.news_popularity_class[nid] for nid in topic_news_ids if self.news_popularity_class[nid] >= 0]
                if len(topic_classes) > 0:
                    topic_class_stats[topic] = {
                        'count': len(topic_news_ids),
                        'class_distribution': np.bincount(topic_classes, minlength=num_classes)
                    }

            print(f'  - Popularity classes computed: {num_classes} classes (quantile_rank, norm_scope={norm_scope})')
            print(f'  - Valid news: {valid_count}')
            print(f'  - Ignored news (0 clicks + padding): {ignored_count}')
            print(f'  - Number of topics: {len(topic_news)}')
            print(f'  - Tie statistics: {tie_stats["total_tie_groups"]} tie groups, {tie_stats["total_ties"]} total tied news')
            print(f'  - Class distribution (valid only): {class_distribution}')

            if len(class_distribution) > 0 and np.sum(class_distribution) > 0:
                class_percentages = (class_distribution / np.sum(class_distribution)) * 100
                min_class_pct = np.min(class_percentages[class_distribution > 0])
                max_class_pct = np.max(class_percentages)
                std_class_pct = np.std(class_percentages[class_distribution > 0])
                print(f'  - Class balance: min={min_class_pct:.2f}%, max={max_class_pct:.2f}%, std={std_class_pct:.2f}%')

            if num_classes <= 10:
                print(f'  - Click count ranges per class:')
                for i, (min_count, max_count) in enumerate(class_click_ranges):
                    count = class_distribution[i] if i < len(class_distribution) else 0
                    print(f'    Class {i}: click_count [{min_count}, {max_count}] -> {count} news')

            if len(topic_class_stats) <= 10:
                print(f'  - Topic-wise class distribution:')
                for topic, stats in sorted(topic_class_stats.items()):
                    print(f'    Topic {topic}: {stats["count"]} news, class_dist={stats["class_distribution"]}')
        else:
            raise ValueError(f"Unknown class_method: {class_method}. Must be 'fixed' or 'quantile_rank'.")

    def _compute_tccm_statistics(self, config: Config):
        """
        Compute statistics required by TCCM (Time and Content-aware Causal Model, CIKM 2023).

        Outputs (stored as attributes and cached to tccm_stats-{dataset}.pkl):
          - self.news_publish_time: [news_num] int64 — hour bucket of each news's first train impression
                                    (-1 if never seen in train; handled as neutral in forward)
          - self.news_entity_indices: [news_num, max_entities] int64 — entity_dict indices per news (0 = pad)
          - self.entity_ctr_table: [num_ctr_buckets, entity_size] int64 — discretized CTR bucket id (0..pop_buckets-1)
          - self.word_ctr_table:   [num_ctr_buckets, vocabulary_size] int64 — same for title words
          - self.tccm_anchor_tsp: int — unix timestamp of train's earliest behavior (for converting dev/test times)

        Reference: TCCM repo `NewsContent.py:208-343`, `DataGeneration.py:86-101`.
        """
        import datetime
        import time as _time

        cache_file = f'tccm_stats-{config.dataset}.pkl'
        # In word_only mode the Model does not register entity buffers, so even when
        # loading from cache we avoid keeping the entity arrays as corpus attributes to reduce RAM usage.
        _cache_use_entity = getattr(config, 'tccm_content_mode', 'word+entity') != 'word_only'
        if os.path.exists(cache_file):
            print(f'Loading TCCM statistics from cache: {cache_file}')
            with open(cache_file, 'rb') as f:
                stats = pickle.load(f)
            self.news_publish_time = stats['news_publish_time']
            self.word_ctr_table = stats['word_ctr_table']
            self.tccm_anchor_tsp = stats['tccm_anchor_tsp']
            self.train_behavior_time_bucket = stats['train_behavior_time_bucket']
            self.dev_behavior_time_bucket = stats['dev_behavior_time_bucket']
            self.test_behavior_time_bucket = stats['test_behavior_time_bucket']
            if _cache_use_entity:
                self.news_entity_indices = stats['news_entity_indices']
                self.entity_ctr_table = stats['entity_ctr_table']
            else:
                # drop heavy entity arrays (news_entity_indices·entity_ctr_table)
                # model.py does not access these attributes in word_only mode afterwards.
                self.news_entity_indices = None
                self.entity_ctr_table = None
            # release the local dict ASAP to let the entity arrays be GC'd
            del stats
            print(f'  news_publish_time   : {self.news_publish_time.shape}')
            if _cache_use_entity:
                print(f'  news_entity_indices : {self.news_entity_indices.shape}')
                print(f'  entity_ctr_table    : {self.entity_ctr_table.shape}')
            else:
                print(f'  entity arrays skipped (tccm_content_mode=word_only)')
            print(f'  word_ctr_table      : {self.word_ctr_table.shape}')
            self._maybe_override_publish_time(config)
            return

        print('Computing TCCM statistics (publish_time, entity/word CTR tables)...')

        def _parse_time(time_str):
            """MIND format 'MM/DD/YYYY HH:MM:SS AM/PM' → unix timestamp (int seconds).
            Returns None on parse failure (e.g. Adressa's numeric timestamps).
            """
            try:
                dt = datetime.datetime.strptime(time_str.strip(), '%m/%d/%Y %I:%M:%S %p')
                return int(_time.mktime(dt.timetuple()))
            except (ValueError, TypeError):
                try:
                    return int(float(time_str.strip()))
                except (ValueError, TypeError):
                    return None

        train_behaviors_path = os.path.join(config.train_root, 'behaviors_raw.tsv')

        # ===== Pass 1: determine anchor (min) and max timestamp on train =====
        anchor_tsp, max_tsp = None, None
        with open(train_behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue
                tsp = _parse_time(parts[2])
                if tsp is None:
                    continue
                if anchor_tsp is None or tsp < anchor_tsp:
                    anchor_tsp = tsp
                if max_tsp is None or tsp > max_tsp:
                    max_tsp = tsp
        if anchor_tsp is None:
            raise RuntimeError('TCCM: failed to parse any train behavior timestamp.')
        self.tccm_anchor_tsp = int(anchor_tsp)
        max_hour_bucket = max(0, (max_tsp - anchor_tsp) // 3600)

        def _to_hour_bucket(tsp):
            return max(0, (tsp - anchor_tsp) // 3600)

        # ===== Entity usage flag =====
        # In word_only mode the Model neither registers entity buffers nor takes the
        # entity branch in forward, so we skip the statistics computation to reduce startup cost.
        # (The entity keys in the pkl are stored as zero arrays for shape compatibility.)
        use_entity = getattr(config, 'tccm_content_mode', 'word+entity') != 'word_only'

        # ===== Step A: news_entity_indices from news_raw.tsv (title_entities field) =====
        max_entities = config.tccm_max_entities
        news_entity_indices = np.zeros((self.news_num, max_entities), dtype=np.int64)
        if use_entity:
            seen_news = set()
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                news_path = os.path.join(prefix, 'news_raw.tsv')
                if not os.path.exists(news_path):
                    continue
                with open(news_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.split('\t')
                        if len(parts) < 7:
                            continue
                        news_id = parts[0]
                        if news_id not in self.news_ID_dict or news_id in seen_news:
                            continue
                        seen_news.add(news_id)
                        news_idx = self.news_ID_dict[news_id]
                        title_entities_str = parts[6]
                        if not title_entities_str.strip():
                            continue
                        try:
                            entities_json = json.loads(title_entities_str)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        slot = 0
                        for ent in entities_json:
                            wiki_id = ent.get('WikidataId', '') if isinstance(ent, dict) else ''
                            if wiki_id and wiki_id in self.entity_dict:
                                news_entity_indices[news_idx, slot] = self.entity_dict[wiki_id]
                                slot += 1
                                if slot >= max_entities:
                                    break
        self.news_entity_indices = news_entity_indices

        # ===== Step B: prepare CTR accumulators =====
        ctr_window_hours = max(1, int(config.tccm_ctr_window_hours))
        pop_buckets = int(config.tccm_pop_buckets)
        num_ctr_buckets = (max_hour_bucket // ctr_window_hours) + 2
        # word_only: do not create entity accumulators and skip the entity loop in Pass 2 as well
        if use_entity:
            entity_click = np.zeros((num_ctr_buckets, config.entity_size), dtype=np.float32)
            entity_expo  = np.zeros((num_ctr_buckets, config.entity_size), dtype=np.float32)
        else:
            entity_click = None
            entity_expo  = None
        word_click   = np.zeros((num_ctr_buckets, config.vocabulary_size), dtype=np.float32)
        word_expo    = np.zeros((num_ctr_buckets, config.vocabulary_size), dtype=np.float32)

        # ===== Step C: publish_time + CTR counts (single pass over train behaviors) =====
        news_publish_time = np.full(self.news_num, -1, dtype=np.int64)
        with open(train_behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) < 5:
                    continue
                tsp = _parse_time(parts[2])
                if tsp is None:
                    continue
                hour_bucket = _to_hour_bucket(tsp)
                ctr_bucket = min(hour_bucket // ctr_window_hours, num_ctr_buckets - 1)
                impressions = parts[4].strip()
                for imp in impressions.split(' '):
                    if len(imp) < 3 or imp[-2] != '-':
                        continue
                    clicked = (imp[-1] == '1')
                    news_id_str = imp[:-2]
                    if news_id_str not in self.news_ID_dict:
                        continue
                    news_idx = self.news_ID_dict[news_id_str]
                    # publish_time: first appearance
                    if news_publish_time[news_idx] == -1 or hour_bucket < news_publish_time[news_idx]:
                        news_publish_time[news_idx] = hour_bucket
                    # Entity CTR — skipped in word_only mode
                    if use_entity:
                        ent_ids = self.news_entity_indices[news_idx]
                        for eid in ent_ids:
                            if eid > 0:
                                entity_expo[ctr_bucket, eid] += 1.0
                                if clicked:
                                    entity_click[ctr_bucket, eid] += 1.0
                    # Word CTR (title words only)
                    title_words = self.news_title_text[news_idx]
                    for wid in title_words:
                        if wid > 1:  # skip <PAD>=0, <UNK>=1
                            word_expo[ctr_bucket, wid] += 1.0
                            if clicked:
                                word_click[ctr_bucket, wid] += 1.0
        self.news_publish_time = news_publish_time

        # ===== Step D: discretize CTR to bucket ids =====
        if use_entity:
            entity_ctr = entity_click / (entity_expo + 0.01)
            entity_ctr[:, 0] = 0.0  # <PAD>
            entity_ctr_bucket = np.ceil(entity_ctr * pop_buckets).astype(np.int64)
            entity_ctr_bucket = np.clip(entity_ctr_bucket, 0, pop_buckets - 1)
        else:
            # Keep the shape for pkl compatibility but store zeros (unused since the Model does not register the buffer)
            entity_ctr_bucket = np.zeros((num_ctr_buckets, config.entity_size), dtype=np.int64)
        word_ctr = word_click / (word_expo + 0.01)
        word_ctr[:, 0] = 0.0  # <PAD>
        word_ctr_bucket = np.ceil(word_ctr * pop_buckets).astype(np.int64)
        word_ctr_bucket = np.clip(word_ctr_bucket, 0, pop_buckets - 1)
        self.entity_ctr_table = entity_ctr_bucket
        self.word_ctr_table = word_ctr_bucket

        print(f'  news_publish_time   : {self.news_publish_time.shape}, '
              f'valid (in-train) = {(news_publish_time >= 0).sum()}/{self.news_num}')
        if use_entity:
            print(f'  news_entity_indices : {self.news_entity_indices.shape}, '
                  f'non-zero ratio = {(news_entity_indices > 0).sum() / news_entity_indices.size:.4f}')
            print(f'  entity_ctr_table    : {self.entity_ctr_table.shape}, '
                  f'mean bucket = {self.entity_ctr_table.mean():.2f}, max = {self.entity_ctr_table.max()}')
        else:
            print(f'  entity stats skipped (tccm_content_mode=word_only)')
        print(f'  word_ctr_table      : {self.word_ctr_table.shape}, '
              f'mean bucket = {self.word_ctr_table.mean():.2f}, max = {self.word_ctr_table.max()}')

        # ===== Step E: per-behavior-line time bucket arrays (for Dataset lookup) =====
        # Each behaviors_raw.tsv line corresponds to one behavior_index.
        # The behavior_index is stored in train_behaviors[5] / dev_behaviors[4] / test_behaviors[4].
        def _collect_behavior_time_buckets(path):
            buckets = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split('\t')
                    if len(parts) < 3:
                        buckets.append(0)
                        continue
                    tsp = _parse_time(parts[2])
                    if tsp is None:
                        buckets.append(0)
                    else:
                        buckets.append(_to_hour_bucket(tsp))
            return np.array(buckets, dtype=np.int64)

        self.train_behavior_time_bucket = _collect_behavior_time_buckets(train_behaviors_path)
        self.dev_behavior_time_bucket = _collect_behavior_time_buckets(os.path.join(config.dev_root, 'behaviors_raw.tsv'))
        self.test_behavior_time_bucket = _collect_behavior_time_buckets(os.path.join(config.test_root, 'behaviors_raw.tsv'))

        print(f'  behavior time buckets: train={self.train_behavior_time_bucket.shape}, '
              f'dev={self.dev_behavior_time_bucket.shape}, test={self.test_behavior_time_bucket.shape}')

        with open(cache_file, 'wb') as f:
            pickle.dump({
                'news_publish_time': self.news_publish_time,
                'news_entity_indices': self.news_entity_indices,
                'entity_ctr_table': self.entity_ctr_table,
                'word_ctr_table': self.word_ctr_table,
                'tccm_anchor_tsp': self.tccm_anchor_tsp,
                'train_behavior_time_bucket': self.train_behavior_time_bucket,
                'dev_behavior_time_bucket': self.dev_behavior_time_bucket,
                'test_behavior_time_bucket': self.test_behavior_time_bucket,
            }, f)
        print(f'TCCM statistics cached to {cache_file}')
        self._maybe_override_publish_time(config)

    def _maybe_override_publish_time(self, config: Config):
        """Replace `news_publish_time` with real article published_time (EB-NERD only).

        Triggered by --tccm_use_real_publish_time. The override pkl is built by
        build_ebnerd_publish_time.py and stores hour-bucket values aligned to the
        same `tccm_anchor_tsp` used here (sanity-checked).

        Also exposes:
          - self.publish_time_valid : np.bool_ array — used by Model to mask unknown publish_time
          - self.tccm_real_publish_time : True flag, picked up by Model to switch elapsed to day units
        """
        self.tccm_real_publish_time = False
        self.publish_time_valid = None
        if not getattr(config, 'tccm_use_real_publish_time', False):
            return
        if config.dataset != 'eb-nerd':
            print(f'[TCCM] --tccm_use_real_publish_time is only implemented for eb-nerd '
                  f'(current dataset={config.dataset}); ignored.')
            return
        override_path = f'news_publish_time_real-{config.dataset}.pkl'
        if not os.path.exists(override_path):
            raise FileNotFoundError(
                f'[TCCM] {override_path} not found. Run `python build_ebnerd_publish_time.py` first.')
        with open(override_path, 'rb') as f:
            ov = pickle.load(f)
        if int(ov['tccm_anchor_tsp']) != int(self.tccm_anchor_tsp):
            raise ValueError(
                f'[TCCM] anchor_tsp mismatch: corpus={self.tccm_anchor_tsp}, '
                f'override={ov["tccm_anchor_tsp"]}. Rebuild the override pkl.')
        if ov['news_publish_time'].shape[0] != self.news_num:
            raise ValueError(
                f'[TCCM] news_num mismatch: corpus={self.news_num}, '
                f'override={ov["news_publish_time"].shape[0]}.')
        self.news_publish_time = ov['news_publish_time'].astype(np.int64)
        self.publish_time_valid = ov['publish_time_valid'].astype(np.bool_)
        self.tccm_real_publish_time = True
        valid = int(self.publish_time_valid.sum())
        pos_pub = self.news_publish_time[self.publish_time_valid]
        pre = int((pos_pub < 0).sum())
        print(f'[TCCM] Real publish_time loaded from {override_path}')
        print(f'       valid={valid}/{self.news_num}, pre-anchor(negative)={pre}, '
              f'range=[{pos_pub.min()}, {pos_pub.max()}] hours')
        print(f'       elapsed will be computed in DAY units (// 24) to cover ~5.5 years '
              f'with tccm_time_buckets={config.tccm_time_buckets}.')

    def _preprocess_for_plm(self, config: Config):
        """Tokenize and store news texts for the PLM."""
        from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
        import pickle
        from tqdm import tqdm

        print('Preprocessing news texts for PLM...')

        # 1. Load PLM tokenizer
        if config.plm_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')
        elif config.plm_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')

        # 2. Cache file path
        plm_title_file = f'plm_title_{config.plm_type}_{config.dataset}.pkl'

        if not os.path.exists(plm_title_file):
            news_plm_title_ids = []
            news_plm_title_masks = []

            for news_index in tqdm(range(self.news_num)):
                title_text = self.news_title_texts[news_index]

                # PLM tokenization
                encoding = tokenizer(
                    title_text,
                    max_length=config.max_title_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='np'
                )

                news_plm_title_ids.append(encoding['input_ids'][0])
                news_plm_title_masks.append(encoding['attention_mask'][0])

            # Convert to NumPy arrays
            self.news_plm_title_ids = np.array(news_plm_title_ids, dtype=np.int32)
            self.news_plm_title_masks = np.array(news_plm_title_masks, dtype=np.int32)

            # Save
            with open(plm_title_file, 'wb') as f:
                pickle.dump({
                    'title_ids': self.news_plm_title_ids,
                    'title_masks': self.news_plm_title_masks
                }, f)
        else:
            # Load
            with open(plm_title_file, 'rb') as f:
                data = pickle.load(f)
                self.news_plm_title_ids = data['title_ids']
                self.news_plm_title_masks = data['title_masks']
        
        print(f'PLM preprocessing completed. Replacing news_title_text with PLM tokenized data.')
        # Replace the default data with PLM data (keeps compatibility with the existing Dataset class)
        self.news_title_text = self.news_plm_title_ids
        self.news_title_mask = self.news_plm_title_masks

        print(f'PLM tokenization completed. Shape: {self.news_plm_title_ids.shape}')