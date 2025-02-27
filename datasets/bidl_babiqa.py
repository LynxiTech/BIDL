# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import os
import re
from functools import reduce
from pprint import pprint

import numpy as np
import six

from datasets.base_dataset import BaseDataset

class BabiQa(BaseDataset):
    """
    https://research.fb.com/downloads/babi/
    """
    CLASSES = None
    WORD_IDX = None
    STORY_MAXLEN = None
    QUERY_MAXLEN = None

    @staticmethod
    def prepare(data_prefix, ann_file):  # data_prefix='data/babiqa/tasks_1-20_v1-2/en/'
        if 'train' in ann_file:
            another = ann_file.replace('train', 'test')
        elif 'test' in ann_file:
            another = ann_file.replace('test', 'train')
        else:
            raise NotImplementedError

        temp = []
        print(f'Using annotation files `{ann_file}` & `{another}` to initialize `vocab`, `word_idx`, etc.')
        for file in [os.path.join(data_prefix, ann_file), os.path.join(data_prefix, another)]:
            fd = open(file, 'rb')
            set_ = get_stories(fd)
            fd.close()
            temp += set_

        vocab = set()
        for story, q, answer in temp:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        story_maxlen = max(map(len, (x for x, _, _ in temp)))
        query_maxlen = max(map(len, (x for _, x, _ in temp)))

        return vocab, word_idx, story_maxlen, query_maxlen

    def load_annotations(self):
        C = BabiQa
        if None in [C.CLASSES, C.WORD_IDX, C.STORY_MAXLEN, C.QUERY_MAXLEN]:
            C.CLASSES, C.WORD_IDX, C.STORY_MAXLEN, C.QUERY_MAXLEN = self.prepare(self.data_prefix, self.ann_file)
            C.CLASSES.insert(0, None)  # XXX

        with open(os.path.join(self.data_prefix, self.ann_file), 'rb') as fd:
            task = get_stories(fd)
        stories, queries, answers = vectorize_stories(task, C.WORD_IDX, C.STORY_MAXLEN, C.QUERY_MAXLEN)

        data_infos = []
        for story, query, answer in zip(stories, queries, answers):
            assert len(story.shape) == len(query.shape) == 1
            info = {
                'img': np.concatenate([story, query], 0).astype('int64'),
                'gt_label': np.array(answer).astype('int64')
            }
            data_infos.append(info)

        return data_infos

    def wrap_text(self, text_content):
        """For Flask demo."""
        raise NotImplementedError


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs, xqs, ys = [], [], []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = word_idx[answer]
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    xs2 = pad_sequences(xs, maxlen=story_maxlen)
    xqs2 = pad_sequences(xqs, maxlen=query_maxlen)
    return xs2, xqs2, np.array(ys)


def tokenize(sent):
    """Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    return [x.strip() for x in re.split(r'(\W+)', sent) if x.strip()]


def get_stories(f, only_supporting=False, max_length=None):
    """Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
        if not max_length or len(flatten(story)) < max_length]
    return data


def parse_stories(lines, only_supporting=False):
    """Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    """
    data, story = [], []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            if only_supporting:  # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:  # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError(f'`sequences` must be a list of iterables. Found non-iterable: {x}')

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: {value}\nYou should set `dtype=object` for variable length strings.")

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f'Shape of sample {trunc.shape[1:]} of sequence at position {idx} is different from expected shape {sample_shape}')
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')

    return x


########################################################################################################################


def main():
    """
    Count some key statistics of bAbIQA before training.
    Should be executed in the project home directory.
    """
    data_prefix = './data/babiqa/en/'
    fns = os.listdir(data_prefix)
    # fns.sort(key=lambda _: int(_[2:_.index('_')]))
    stat_dict = {}
    for ann_file in fns:  
        if 'test' in ann_file:
            continue
        vocab, word_idx, story_maxlen, query_maxlen = BabiQa.prepare(data_prefix, ann_file)
        vocab_size = len(vocab) + 1
        stat_dict[ann_file] = [vocab_size, story_maxlen, query_maxlen]
    pprint(stat_dict)


if __name__ == '__main__':
    main()
