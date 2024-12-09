import torch
from torch.utils.data import Dataset

COLUMNS = ["TOKEN",
           "NE-COARSE-LIT",
           "NE-COARSE-METO",
           "NE-FINE-LIT",
           "NE-FINE-METO",
           "NE-FINE-COMP",
           "NE-NESTED",
           "NEL-LIT",
           "NEL-METO",
           "MISC"]


def _read_conll(path, encoding='utf-8', sep=None, indexes=None, dropna=True):
    r"""
    Construct a generator to read conll items.
    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        # import pdb;pdb.set_trace()
        sample = [sample[i] for i in indexes]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample



    with open(path, 'r', encoding=encoding) as f:
        sample = []
        start = next(f).strip()

        data = []
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if 'NE-COARSE-LIT' in line:
                continue
            if 'DOCSTART' in line:
                continue
            if '### ' in line:
                continue
            if "# id" in line:
                continue
            if "# " in line:
                continue
            if "Token" in line:
                continue
            if "TOKEN" in line:
                continue
            if line == '':
                if len(sample):
                    try:

                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                            continue
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            elif 'EndOfSentence' in line:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])

                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                            continue
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            else:

                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

        if len(sample) > 0:
            try:

                res = parse_conll(sample)
                if ['TOKEN'] not in res:
                    if ['Token'] not in res:
                        data.append([line_idx, res])
            except Exception as e:
                if dropna: # TODO: dangerous this thing here
                    return
                print('Invalid instance ends at line: {}'.format(line_idx))
                raise e

        return data


class NewsDataset(Dataset):
    """
    """

    def __init__(self, tsv_dataset, tokenizer,
                 max_len,
                 test=False,
                 nerc_coarse_label_map={},
                 nerc_fine_label_map={}):
        """
        Initiliazes a dataset in IOB format.
        :param tsv_dataset: tsv filename of the train/test/dev dataset
        :param tokenizer: the LM tokenizer
        :param max_len: the maximum sequence length, get be 512 for BERT-based LMs
        :param test: if it is the test dataset or not - can be disconsidered for now
        :param label_map: the label map {0: 'B-pers'}
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test = test

        indexes = list(range(len(COLUMNS)))

        self.tsv_dataset = tsv_dataset

        COARSE = 3
        FINE = 5

        self.phrases = _read_conll(self.tsv_dataset, encoding='utf-8', sep='\t', indexes=indexes, dropna=True)

        self.tokens = [item[1][0] for item in self.phrases]

        # TODO: this is a hack to get two types of labels
        self.nerc_coarse_targets = [item[1][COARSE] for item in self.phrases]
        self.nerc_fine_targets = [item[1][FINE] for item in self.phrases]

        ## ----------------- COARSE ----------------- ##

        self.nerc_coarse_label_map = nerc_coarse_label_map
        unique_token_labels = set(sum(self.nerc_coarse_targets, []))
        label_mapped = dict(
            zip(unique_token_labels, range(len(unique_token_labels))))
        missed_labels = set(label_mapped) - set(nerc_coarse_label_map)

        print("Appended following labels to label_map:", missed_labels)

        nerc_coarse_num_labels = len(self.nerc_coarse_label_map)
        for i, missed_label in enumerate(missed_labels):
            self.nerc_coarse_label_map[missed_label] = nerc_coarse_num_labels + i

        self.nerc_coarse_token_targets = [[self.nerc_coarse_label_map[element]
                                            for element in item[1][COARSE]] for item in self.phrases]

        ## ----------------- FINE --------------------- ##

        self.nerc_fine_label_map = nerc_fine_label_map
        unique_token_labels = set(sum(self.nerc_fine_targets, []))
        label_mapped = dict(
            zip(unique_token_labels, range(len(unique_token_labels))))
        missed_labels = set(label_mapped) - set(nerc_fine_label_map)

        print("Appended following labels to label_map:", missed_labels)

        nerc_fine_num_labels = len(self.nerc_fine_label_map)
        for i, missed_label in enumerate(missed_labels):
            self.nerc_fine_label_map[missed_label] = nerc_fine_num_labels + i

        self.nerc_fine_token_targets = [[self.nerc_fine_label_map[element]
                                           for element in item[1][FINE]] for item in self.phrases]

    def __len__(self):
        return len(self.phrases)

    def get_filename(self):
        return self.tsv_dataset

    def get_nerc_coarse_label_map(self):
        return self.nerc_coarse_label_map
    def get_nerc_fine_label_map(self):
        return self.nerc_fine_label_map

    def get_inverse_nerc_coarse_label_map(self):
        return {v: k for k, v in self.get_nerc_coarse_label_map.items()}
    def get_inverse_nerc_fine_label_map(self):
        return {v: k for k, v in self.get_nerc_fine_label_map.items()}
    def tokenize_and_align_labels(self, sequence, nerc_coarse_token_targets, nerc_fine_token_targets):
        """
        :param sequence:
        :param token_targets:
        :return:
        """
        tokenized_inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word).
            is_split_into_words=True,
            return_token_type_ids=True
        )
        labels_coarse = []
        labels_fine = []
        label_all_tokens = False
        # for i, label in enumerate(tokens):
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None

        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                labels_coarse.append(-100)
                labels_fine.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                labels_coarse.append(nerc_coarse_token_targets[word_idx])
                labels_fine.append(nerc_fine_token_targets[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    labels_coarse.append(nerc_coarse_token_targets[word_idx])
                    labels_fine.append(nerc_fine_token_targets[word_idx])
                else:
                    labels_coarse.append(-100)
                    labels_fine.append(-100)

            previous_word_idx = word_idx

        tokenized_inputs["token_coarse_targets"] = labels_coarse
        tokenized_inputs["token_fine_targets"] = labels_fine
        return tokenized_inputs

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        sequence = self.tokens[index]
        nerc_coarse_token_targets = self.nerc_coarse_token_targets[index]
        nerc_fine_token_targets = self.nerc_fine_token_targets[index]

        encoding = self.tokenize_and_align_labels(sequence, nerc_coarse_token_targets, nerc_fine_token_targets)

        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        token_type_ids = torch.tensor(
            encoding['token_type_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)

        token_coarse_targets = torch.tensor(
            encoding['token_coarse_targets'], dtype=torch.long)
        token_fine_targets = torch.tensor(
            encoding['token_fine_targets'], dtype=torch.long)

        assert input_ids.shape == attention_mask.shape
        assert token_coarse_targets.shape == input_ids.shape
        assert token_fine_targets.shape == input_ids.shape

        return {
            'sequence': ' '.join(sequence),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_coarse_targets': token_coarse_targets,
            'token_fine_targets': token_fine_targets,
            'token_type_ids': token_type_ids}

    def get_num_coarse_token_labels(self):
        """
        :return:
        """
        return len(self.nerc_coarse_label_map)

    def get_num_fine_token_labels(self):
        """
        :return:
        """
        return len(self.nerc_fine_label_map)