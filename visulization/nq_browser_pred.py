# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Web frontend for browsing Google's Natural Questions.
Example usage:
pip install absl-py
pip install jinja2
pip install tornado==5.1.1
pip install wsgiref
python nq_browser --nq_jsonl=nq-train-sample.jsonl.gz
python nq_browser --nq_jsonl=nq-dev-sample.jsonl.gz --dataset=dev --port=8081
"""

import os
import gzip
import json
import wsgiref.simple_server
from absl import app
from absl import flags
import jinja2
import numpy as np
import tornado.web
import tornado.wsgi
import logging

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

FLAGS = flags.FLAGS

flags.DEFINE_string('nq_jsonl', "../input/tensorflow2-question-answering/simplified-nq-dev.jsonl",
                    'Path to jsonlines file containing Natural Questions.')
flags.DEFINE_string('pred_json', "../output/models/albert-xxlarge-tfidf-600-top8-V0/predictions99998.json",
                    'Path to json file containing predictions.')
flags.DEFINE_boolean('gzipped', False, 'Whether the jsonlines are gzipped.')
flags.DEFINE_enum('dataset', 'dev', ['train', 'dev'],
                  'Whether this is training data or dev data.')
flags.DEFINE_integer('port', 8080, 'Port to listen on.')
flags.DEFINE_integer('max_examples', 1600,
                     'Max number of examples to load in the browser.')
flags.DEFINE_enum('mode', 'all_examples',
                  ['all_examples', 'long_answers', 'short_answers'],
                  'Subset of examples to show.')


class LongAnswerCandidate(object):
    """Representation of long answer candidate."""

    def __init__(self, contents, index, is_answer, contains_answer, start_token, end_token, top_level):
        self.contents = contents
        self.index = index
        self.is_answer = is_answer
        self.contains_answer = contains_answer
        self.start_token = start_token
        self.end_token = end_token
        self.top_level = top_level
        if is_answer:
            self.style = 'is_answer'
        elif contains_answer:
            self.style = 'contains_answer'
        else:
            self.style = 'not_answer'


class Example(object):
    """Example representation."""

    def __init__(self, json_example):
        self.json_example = json_example

        # Whole example info.
        self.url = json_example['document_url']
        self.title = json_example.get('document_title', 'Wikipedia')
        self.example_id = str(self.json_example['example_id'])
        self.document_html = self.json_example['document_text']
        self.document_tokens = self.document_html.split()
        self.question_text = json_example['question_text']
        self.pred_long_answer_score = None
        self.pred_long_answer_start_token = None
        self.pred_long_answer_end_token = None
        self.pred_short_answers_start_token = None
        self.pred_short_answers_end_token = None
        self.pred_yes_no_answer = None
        self.pred_short_answers_score = None
        self.short_answer_pred_text = None

        if FLAGS.dataset == 'train':
            if len(json_example['annotations']) != 1:
                raise ValueError(
                    'Train set json_examples should have a single annotation.')
            annotation = json_example['annotations'][0]
            self.has_long_answer = annotation['long_answer']['start_token'] >= 0
            self.has_short_answer = annotation[
                                        'short_answers'] or annotation['yes_no_answer'] != 'NONE'

        elif FLAGS.dataset == 'dev':
            if len(json_example['annotations']) != 5:
                raise ValueError('Dev set json_examples should have five annotations.')
            self.has_long_answer = sum([
                annotation['long_answer']['start_token'] >= 0
                for annotation in json_example['annotations']
            ]) >= 2
            self.has_short_answer = sum([
                bool(annotation['short_answers']) or
                annotation['yes_no_answer'] != 'NONE'
                for annotation in json_example['annotations']
            ]) >= 2

        self.long_answers = [
            a['long_answer']
            for a in json_example['annotations']
            if a['long_answer']['start_token'] >= 0 and self.has_long_answer
        ]
        self.short_answers = [
            a['short_answers']
            for a in json_example['annotations']
            if a['short_answers'] and self.has_short_answer
        ]
        self.yes_no_answers = [
            a['yes_no_answer']
            for a in json_example['annotations']
            if a['yes_no_answer'] != 'NONE' and self.has_short_answer
        ]

        if self.has_long_answer:
            long_answer_bounds = [
                (la['start_token'], la['end_token']) for la in self.long_answers
            ]
            long_answer_counts = [
                long_answer_bounds.count(la) for la in long_answer_bounds
            ]
            long_answer = self.long_answers[np.argmax(long_answer_counts)]
            self.long_answer_text = self.render_long_answer(long_answer)
            # exp
            self.long_answer_start = long_answer['start_token']
            self.long_answer_end = long_answer['end_token']
            # endexp

        else:
            self.long_answer_text = ''

        if self.has_short_answer:
            short_answers_ids = [[
                (s['start_token'], s['end_token']) for s in a
            ] for a in self.short_answers] + [a for a in self.yes_no_answers]
            short_answers_counts = [
                short_answers_ids.count(a) for a in short_answers_ids
            ]

            self.short_answers_texts = [
                ', '.join([str(s['start_token']) + ':' + str(s['end_token']) + " " +
                           self.render_span(s['start_token'], s['end_token'])
                           for s in short_answer
                           ])
                for short_answer in self.short_answers
            ]

            self.short_answers_texts += self.yes_no_answers
            self.short_answers_text = self.short_answers_texts[np.argmax(
                short_answers_counts)]
            self.short_answers_texts = set(self.short_answers_texts)

        else:
            self.short_answers_texts = []
            self.short_answers_text = ''

        self.candidates = self.get_candidates(
            self.json_example['long_answer_candidates'])

        self.candidates_with_answer = [
            i for i, c in enumerate(self.candidates) if c.contains_answer
        ]

    def render_long_answer(self, long_answer):
        """Wrap table rows and list items, and render the long answer.
        Args:
          long_answer: Long answer dictionary.
        Returns:
          String representation of the long answer span.
        """

        if long_answer['end_token'] - long_answer['start_token'] > 500:
            return 'Large long answer'

        html_tag = self.document_tokens[long_answer['end_token'] - 1]
        if html_tag == '</Table>' and self.render_span(
                long_answer['start_token'], long_answer['end_token']).count('<TR>') > 30:
            return 'Large table long answer'

        elif html_tag == '</Tr>':
            return '<TABLE>{}</TABLE>'.format(
                self.render_span(long_answer['start_token'], long_answer['end_token']))

        elif html_tag in ['</Li>', '</Dd>', '</Dd>']:
            return '<Ul>{}</Ul>'.format(
                self.render_span(long_answer['start_token'], long_answer['end_token']))

        else:
            return self.render_span(long_answer['start_token'],
                                    long_answer['end_token'])

    def render_span(self, start, end):
        return " ".join(self.document_tokens[start:end])

    def get_candidates(self, json_candidates, keep_top_level_only=True):
        """Returns a list of `LongAnswerCandidate` objects for top level candidates.
        Args:
          json_candidates: List of Json records representing candidates.
          keep_top_level_only: bool
        Returns:
          List of `LongAnswerCandidate` objects.
        """
        candidates = []
        if keep_top_level_only:
            top_level_candidates = [c for c in json_candidates if c['top_level']]
        else:
            top_level_candidates = json_candidates
        for candidate in top_level_candidates:
            tokenized_contents = ' '.join([
                t for t in self.document_tokens
                [candidate['start_token']:candidate['end_token']]
            ])

            start = candidate['start_token']
            end = candidate['end_token']
            is_answer = self.has_long_answer and np.any(
                [(start == ans['start_token']) and (end == ans['end_token'])
                 for ans in self.long_answers])
            contains_answer = self.has_long_answer and np.any(
                [(start <= ans['start_token']) and (end >= ans['end_token'])
                 for ans in self.long_answers])

            start_token = candidate['start_token']
            end_token = candidate['end_token']
            top_level = candidate["top_level"]

            candidates.append(
                LongAnswerCandidate(tokenized_contents, len(candidates), is_answer,
                                    contains_answer, start_token, end_token, top_level))

        return candidates


class Prediction(object):
    """Prediction representation."""

    def __init__(self, json_example):
        """
        E.g.
        {
            "example_id": 2689040320115501056,
            "long_answer_score": -0.8586194217205048,
            "long_answer": {
                "end_token": 601,
                "start_token": 524,
                "start_token": -1,
                "end_token": -1
            },
            "short_answers": [
                {
                    "end_token": 528,
                    "start_token": 525,
                    "start_token": -1,
                    "end_token": -1
                }
            ],
            "yes_no_answer": "NONE",
            "short_answers_score": -0.8586194217205048
        },

        Args:
            json_example: dict
        """
        self.json_example = json_example

        # Whole example info.
        self.example_id = str(self.json_example['example_id'])
        self.long_answer_score = json_example['long_answer_score']
        self.long_answer_start_token = json_example['long_answer']['start_token']
        self.long_answer_end_token = json_example['long_answer']['end_token']
        self.short_answers_start_token = json_example['short_answers'][0]['start_token']
        self.short_answers_end_token = json_example['short_answers'][0]['end_token']
        self.yes_no_answer = json_example['yes_no_answer']
        self.short_answers_score = json_example['short_answers_score']


def has_long_answer(json_example):
    for annotation in json_example['annotations']:
        if annotation['long_answer']['start_token'] >= 0:
            return True
    return False


def has_short_answer(json_example):
    for annotation in json_example['annotations']:
        if annotation['short_answers']:
            return True
    return False


def load_examples(fileobj):
    """Reads jsonlines containing NQ examples.
    Args:
      fileobj: File object containing NQ examples.
    Returns:
      Dictionary mapping example id to `Example` object.
    """

    def _load(examples, f):
        """Read serialized json from `f`, create examples, and add to `examples`."""

        for l in f:
            json_example = json.loads(l)
            if FLAGS.mode == 'long_answers' and not has_long_answer(json_example):
                continue

            elif FLAGS.mode == 'short_answers' and not has_short_answer(json_example):
                continue

            example = Example(json_example)
            examples[example.example_id] = example

            if len(examples) == FLAGS.max_examples:
                break

    examples = {}
    if FLAGS.gzipped:
        _load(examples, gzip.GzipFile(fileobj=fileobj))
    else:
        _load(examples, fileobj)

    return examples


def load_predictions(fileobj):
    """Reads json containing predictions
    Args:
      fileobj: File object containing predictions.
    Returns:
      Dictionary mapping example id to `Prediction` object.
    """

    def _load(predictions, f):
        """Read serialized json from `f`, create examples, and add to `examples`."""

        # with open(f) as json_file:
        data = json.load(f)
        for p in data['predictions']:
            prediction = Prediction(p)
            predictions[prediction.example_id] = prediction

    predictions = {}
    _load(predictions, fileobj)

    return predictions


def render_short_answer(tokens, start, end):
    text = " ".join(tokens[start:end])
    return text


def create_expreds(examples, predictions):
    """
    Args:
        examples: dict
        predictions: dict

    Returns:
        expreds: dict
        examples: dict
    """
    expreds = examples

    for example in expreds:
        if example in predictions:
            # # mark up predicted long answer for highlighting
            pred = predictions[example]
            for cand in examples[example].candidates:
                if cand.start_token == pred.long_answer_start_token and cand.end_token == pred.long_answer_end_token:
                    if cand.is_answer:
                        # ## tp means pred == gold
                        cand.style = "tp"
                    else:
                        # ## pred_answer means pred != gold
                        cand.style = "pred_answer"

            expreds[example].pred_long_answer_score = predictions[example].long_answer_score
            expreds[example].pred_long_answer_start_token = predictions[example].long_answer_start_token
            expreds[example].pred_long_answer_end_token = predictions[example].long_answer_end_token
            expreds[example].pred_short_answers_start_token = predictions[example].short_answers_start_token
            expreds[example].pred_short_answers_end_token = predictions[example].short_answers_end_token
            expreds[example].pred_yes_no_answer = predictions[example].yes_no_answer
            expreds[example].pred_short_answers_score = predictions[example].short_answers_score
            expreds[example].short_answer_pred_text = render_short_answer(expreds[example].document_tokens, predictions[
                example].short_answers_start_token, predictions[example].short_answers_end_token)

    return expreds, examples


class MainHandler(tornado.web.RequestHandler):
    """Displays an overview table of the loaded NQ examples."""

    def initialize(self, jinja2_env, examples):
        self.env = jinja2_env
        self.tmpl = self.env.get_template('index.html')
        self.examples = examples

    def get(self):
        res = self.tmpl.render(
            dataset=FLAGS.dataset.capitalize(), examples=self.examples.values())
        self.write(res)


class ExPredHandler(tornado.web.RequestHandler):
    """Displays an overview table of the loaded NQ examples and predictions."""

    def initialize(self, jinja2_env, expreds):
        self.env = jinja2_env
        self.tmpl = self.env.get_template('expreds.html')
        self.expreds = expreds

    def get(self):
        res = self.tmpl.render(
            dataset=FLAGS.dataset.capitalize(), expreds=self.expreds.values())
        self.write(res)


class HtmlHandler(tornado.web.RequestHandler):
    """Displays the html field contained in a NQ example."""

    def initialize(self, examples):
        self.examples = examples

    def get(self):
        example_id = str(self.get_argument('example_id'))
        self.write(self.examples[example_id].document_html)


class FeaturesHandler(tornado.web.RequestHandler):
    """Displays a detailed view of the features extracted from a NQ example."""

    def initialize(self, jinja2_env, examples):
        self.env = jinja2_env
        self.tmpl = self.env.get_template('features.html')
        self.examples = examples

    def get(self):
        example_id = str(self.get_argument('example_id'))
        res = self.tmpl.render(
            dataset=FLAGS.dataset.capitalize(), example=self.examples[example_id])
        self.write(res)


class PredictionsHandler(tornado.web.RequestHandler):
    """Add predictions html template."""

    def initialize(self, jinja2_env, predictions):
        self.env = jinja2_env
        self.tmpl = self.env.get_template('predictions.html')
        self.predictions = predictions

    def get(self):
        res = self.tmpl.render(
            dataset=FLAGS.dataset.capitalize(), predictions=self.predictions.values())
        self.write(res)


class NqServer(object):
    """Serves all different tools."""

    def __init__(self, web_path, examples, predictions, expreds):
        """
        """
        tmpl_path = web_path + '/templates'
        static_path = web_path + '/static'
        jinja2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(tmpl_path))

        self.application = tornado.wsgi.WSGIApplication([
            (r'/', MainHandler, {
                'jinja2_env': jinja2_env,
                'examples': examples
            }),
            (r'/expreds', ExPredHandler, {
                'jinja2_env': jinja2_env,
                'expreds': expreds
            }),
            (r'/html', HtmlHandler, {
                'examples': examples
            }),
            (r'/features', FeaturesHandler, {
                'jinja2_env': jinja2_env,
                'examples': examples
            }),
            (r'/predictions', PredictionsHandler, {
                'jinja2_env': jinja2_env,
                'predictions': predictions
            }),
            (r'/static/(.*)', tornado.web.StaticFileHandler, {
                'path': static_path
            }),
        ])

    def serve(self):
        """Main entry point for the NqSever."""
        logger.info("NqServer listening at port {}".format(FLAGS.port))
        server = wsgiref.simple_server.make_server('', FLAGS.port, self.application)
        server.serve_forever()


def main(unused_argv):
    with open(FLAGS.nq_jsonl, encoding="utf8") as fileobj:
        examples = load_examples(fileobj)

    with open(FLAGS.pred_json) as fileobj:
        predictions = load_predictions(fileobj)

    expreds, examples = create_expreds(examples, predictions)

    web_path = os.path.dirname(os.path.realpath(__file__))
    NqServer(web_path, examples, predictions, expreds).serve()


if __name__ == '__main__':
    app.run(main)
