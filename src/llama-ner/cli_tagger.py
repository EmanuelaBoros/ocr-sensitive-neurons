"""
This module contains functions to predict entities from text using a news agency NER model.

It includes the following functions:
- `send_prediction_request(json_request: List[str], language: str)`: Sends a prediction request to the NER model.
- `predict_entities(content_items: List[Dict[str, Any]])`: Predicts entities for a list of content items.

Date: July 2023
"""

from typing import Dict, List, Optional, Any
import requests
from tqdm import tqdm
import string
import pysbd
import json
import torch
from utils import Timer, chunk
import argparse
import logging
import glob
from dask.diagnostics import ProgressBar
import dask.bag as db
import os
import sys
import time
from dask.distributed import Client, LocalCluster

# The default value might be 1000
# sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000
# get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))
# add the current directory to sys.path
sys.path.insert(0, current_directory)
logger = logging.getLogger(__name__)

SENTENCE_SEGMENTER = {'fr': pysbd.Segmenter(language="fr", clean=False),
                      'de': pysbd.Segmenter(language="de", clean=False)}

WIKIDATA_IDs = {
    "AFP": "Q40464",
    "ANP": "Q966898",
    "ANSA": "Q392934",
    "AP": "Q40469",
    "APA": "Q680662",
    "ATS-SDA": "Q430109",
    "Belga": "Q815453",
    "BTA": "Q2031809",
    "CTK": "Q341118",
    "DDP-DAPD": "Q265330",
    "DNB": "Q1205856",
    "Domei": "Q2913752",
    "DPA": "Q312653",
    "Europapress": "NIL",
    "Extel": "Q1525848",
    "Havas": "Q2826560",
    "Interfax": "Q379271",
    "Kipa": "Q1522416",
    "PAP": "Q1484980",
    "Reuters": "Q130879",
    "SPK-SMP": "Q2256560",
    "Stefani": "Q1415690",
    "TANJUG": "Q371267",
    "TASS": "Q223799",
    "Telunion": "Q3517301",
    "TT": "Q1312158",
    "UP-UPI": "Q493845",
    "Wolff": "Q552226",
    "Xinhua": "Q204839",
}

WIKIDATA_IDs = {k.lower(): v for k, v in dict(WIKIDATA_IDs).items()}
def send_prediction_request(json_request: List[str], language: str) -> List[Dict[str, Any]]:
    """
    Send a prediction request to the API provided by TorchServe the agency NER model for a given language.
    :param json_request: List of JSON objects representing sentences to predict entities for.
    :param language: The language of the sentences.
    :return: List of entities predicted by the NER model.
    """
    url = f'http://127.0.0.1:8080/predictions/agency_{language}'

    headers = {'Content-Type': 'application/json'}
    json_data = json.dumps(json_request)
    response = requests.post(url, data=json_data, headers=headers)

    if not response.status_code == 200:
        print(
            f'Failed to send prediction request. Error code: {response.status_code}')

    results = response.json()
    return results

import re
def remove_space_before_punctuation(text):
    punctuation = re.escape(string.punctuation)
    return re.sub(rf"\s([{punctuation}](?:\s|$))", r'\1', text)


# TODO: add timing for 100 documents
def predict_entities(content_items: List[Dict[str, Any]]) -> List[List[Any]]:
    # """
    # Predict entities for a list of content items using a specific NER model.
    # :param content_items: List of content items to predict entities for.
    # :return: List of entities predicted for each content item.
    # """
    sys.setrecursionlimit(10000)  # Increase the recursion limit to 10000

    timings = []
    torch.cuda.empty_cache()
    timings.append({'load_models': 0.0})

    count = 0
    result_json = []

    # convert the filter object to a list
    content_items_list = list(content_items)

    for ci in tqdm(content_items_list, total=len(content_items_list)):
        count += 1
        timing = {}

        language = ci['lg_comp']
        article = ci['ft']
        if language in ['de', 'fr']:
            # TODO: add timing
            segmenter_start_time = time.time()
            try:
                sentences = SENTENCE_SEGMENTER[language].segment(article)
            except:
                sentences = []
            if len(sentences) > 0:
                # Store the time taken
                timing['segment'] = time.time() - segmenter_start_time
                timing_article = []

                api_requests = []
                for sentence in sentences:
                    api_requests.append(json.dumps(
                        {"text": sentence, "language": language}))

                timing_sentence = {}
                pred_start_time = time.time()
                article_entities = send_prediction_request(
                    api_requests, language)
                timing_sentence['sentence_prediction'] = time.time() - pred_start_time  # Store the time taken

                json_start_time = time.time()
                total_sentence_length = 0

                for sentence_idx, item in enumerate(zip(article_entities, sentences)):
                    entities, sentence = item
                    for entity in entities:
                        if entity[1] != 'O':
                            if entity[0] not in string.punctuation:
                                if len(entity[0]) > 1:
                                    # TODO: character position

                                    original_string, original_label = entity[0], entity[1]
                                    # print('*******', original_string, original_label)
                                    original_string = remove_space_before_punctuation(original_string)

                                    lSentenceOffset = sentence.find(original_string)
                                    rSentenceOffset = lSentenceOffset + len(original_string)

                                    lArticleOffset = total_sentence_length + lSentenceOffset
                                    rArticleOffset = total_sentence_length + rSentenceOffset

                                    # for punctuation in string.punctuation:
                                    #     text = text.replace(punctuation, ' ' + punctuation + ' ')
                                    if 'ATB' in original_label:
                                        label = original_label.replace('ATB', 'ATS').split(".")[-1]
                                        original_label = original_label.replace('ATB', 'ATS')
                                    else:
                                        label = original_label.split(".")[-1]
                                    # print('------label', label, 'original', original_label)
                                    # print('------', original_string, sentence[lSentenceOffset:rSentenceOffset])
                                    wiki_id = 'NIL'
                                    if ('articleauthor' not in label) and ('unk' not in label):
                                        if label.lower() in WIKIDATA_IDs:
                                            wiki_id = WIKIDATA_IDs[label.lower()]
                                        else:
                                            wiki_id = 'NIL'

                                    entity_json = {
                                        "entity": original_label,
                                        "surface": original_string,
                                        "qid": wiki_id,
                                        "lSentenceOffset": lSentenceOffset,
                                        "rSentenceOffset": rSentenceOffset,
                                        "sentence_idx:": sentence_idx,
                                        "lArticleOffset": lArticleOffset,
                                        "rArticleOffset": rArticleOffset,
                                        'id': ci["id"] + f":{sentence_idx}:{lSentenceOffset}:{rSentenceOffset}:{lArticleOffset}:{rArticleOffset}:newsag:bert_{language}"}

                                    # print(json.dumps(entity_json))
                                    result_json.append(entity_json)

                    total_sentence_length += len(sentence) + 1

                    timing_sentence['sentence_result_json'] = time.time(
                    ) - json_start_time  # Store the time taken
                    # Update cumulative offset after processing each sentence
                    timing_article.append(timing_sentence)
                timing['entire_article'] = timing_article
            timings.append(timing)
            # if count > 10:
            #     break

    return result_json, timings


def run_newsagency_tagger(input_dir: str,
                          output_dir: str,
                          prefix: Optional[str] = None) -> None:

    t = Timer()
    total_time_start = time.time()
    timings = []

    if prefix is not None:
        path = f"{input_dir}/{prefix}*.jsonl.bz2"

        output_dir_prefix = os.path.join(output_dir, prefix)
        if not os.path.exists(output_dir_prefix):
            os.mkdir(output_dir_prefix)
    else:
        path = f"{input_dir}/*.jsonl.bz2"

        output_dir_prefix = output_dir

    logger.info(f"Indexing files in {path}")
    file_time_start = time.time()
    files = glob.glob(path)
    logger.info(
        f'Number of files: {len(files)}. Time taken to read files: {time.time() - file_time_start}')

    batches = list(chunk(files, 10))
    total = len(batches)

    with client:
        for i, b in enumerate(batches):
            batch_time_start = time.time()
            timing = {'batch': i}
            logger.info(f'Parsing {i}/{total}: {b}')

            read_time_start = time.time()
            bag_articles = db.read_text(b) \
                .filter(lambda s: len(s) > 2) \
                .map(json.loads) \
                .filter(lambda ci: ci['tp'] in ['ar', 'page'])

            logger.info(
                f'Time taken to read and filter articles: {time.time() - read_time_start}')
            timing['read_and_filter'] = time.time() - read_time_start

            # print(len(list(bag_articles)))

            process_time_start = time.time()

            output = bag_articles.map_partitions(predict_entities).compute()
            # This takes every other element, starting from the first one
            bag_articles = output[::2]
            # This takes every other element, starting from the second one
            timing_articles = output[1::2]
            print(i)
            bag_articles = db.from_sequence(bag_articles)
            bag_mentions = bag_articles.map(json.dumps)

            # print(len(list(bag_mentions)))

            timing['timing_articles'] = [
                json.dumps(article) for article in timing_articles]
            logger.info(
                f'Time taken to process articles: {time.time() - process_time_start}')
            timing['process_articles'] = time.time() - process_time_start

            write_time_start = time.time()

            with ProgressBar():
                bag_mentions.to_textfiles(f'{output_dir_prefix}/' + f'{i}_*.jsonl.bz2',
                    name_function=lambda x: str(x))
        client.close()
        logger.info(
            f'Time taken to write mentions: {time.time() - write_time_start}')
        timing['write_articles'] = time.time() - write_time_start
        logger.info(
            f"Batch time: {t.tick()}. Total time taken for this batch: {time.time() - batch_time_start}")
        timing['batch_time'] = time.time() - batch_time_start

        timings.append(timing)
        with open('batch_timings_dask_ts_batch_8workers10batch_all.json', 'w') as file:
            json.dump(timings, file)

    logger.info(
        f'Total time taken for run_newsagency_tagger: {time.time() - total_time_start}')


def parse_args():
    description = ""
    epilog = ""
    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument(
        "--input_dir",
        dest="input_dir",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output_dir",
        dest="output_dir",
    )

    parser.add_argument(
        "-l",
        "--logfile",
        dest="logfile",
        help="write log to FILE",
        metavar="FILE"
    )

    parser.add_argument(
        "--verbose",
        dest="verbose",
        default=3,
        type=int,
        metavar="LEVEL",
        help="set verbosity level: 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO 4=DEBUG (default %(default)s)",
    )

    parser.add_argument(
        "--prefix",
        dest="prefix",
        # default="000",
        type=str,
        help="todo",
    )

    return parser.parse_args()


if __name__ == "__main__":

    arguments = parse_args()

    log_levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]

    handlers = [logging.StreamHandler()]
    if arguments.logfile:
        handlers.append(logging.FileHandler(arguments.logfile, mode="w"))

    logging.basicConfig(
        level=log_levels[arguments.verbose],
        format="%(asctime)-15s %(levelname)s: %(message)s",
        handlers=handlers,
    )

    n = torch.cuda.device_count()
    # Connect to an existing Dask scheduler

    cluster = LocalCluster(n_workers=arguments.workers)
    client = Client(cluster)

    run_newsagency_tagger(
        arguments.input_dir,
        arguments.output_dir,
        arguments.prefix)
