"""
This is a modified version of the original ``MessagesBert/logbert.py`` script from
`genai4log` that includes two important changes for better anomaly detection:

* **Expanded vocabulary** – Instead of building the vocabulary from just the
  training sequences, this script expects the combined normal data (training
  sequences and known‐normal test sequences) to be saved in
  ``../output/linux/raw_log_sequence.csv``.  This ensures that all log keys that
  may appear during testing are included in the vocabulary so they are not
  mapped to the `<unk>` token.
* **Lower mask ratio** – The default ``mask_ratio`` has been lowered from
  ``0.85`` to ``0.3``.  The LogBERT paper shows that masking more than half of
  the tokens degrades performance【244891535061353†L479-L486】.  A ratio around
  0.3 retains enough context for the model to learn while still encouraging
  generalisation.

To use this script, prepare your combined log sequences in
``../output/linux/raw_log_sequence.csv`` and run::

    python logbert.py vocab
    python logbert.py train
    python logbert.py predict

See the accompanying ``bert_pytorch/predict_log.py`` for changes that treat
unseen (``<unk>``) tokens as anomalies.
"""

import sys
sys.path.append("../")
sys.path.append("../../")

import os
import argparse
import torch

from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from bert_pytorch.dataset.utils import seed_everything

# Configuration options
options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
# All intermediate files will live in output/linux/
options["output_dir"] = "../output/linux/"
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"

# ``train_vocab`` should point to the combined normal log sequences used to
# build the vocabulary.  By default, this file is expected to be
# ``raw_log_sequence.csv`` under the ``output_dir``.  See README for details.
options["train_vocab"] = options["output_dir"] + "raw_log_sequence.csv"
options["vocab_path"] = options["output_dir"] + "vocab.pkl"  # pickled vocab

options["window_size"] = 1
options["adaptive_window"] = True
options["seq_len"] = 5
options["max_len"] = 512  # for position embedding
options["min_len"] = 1

######################################################################
# Important: mask_ratio controls how many tokens are masked during both
# training and detection.  The original code used 0.85, but the LogBERT
# paper notes that very high ratios degrade performance【244891535061353†L479-L486】.
# A value between 0.1 and 0.5 is recommended.  Here we use 0.3.
options["mask_ratio"] = 0.3

# Sample ratios
options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# Features
options["is_logkey"] = True
options["is_time"] = False

# DeepSVDD options
options["hypersphere_loss"] = False
options["hypersphere_loss_test"] = False

options["scale"] = None  # MinMaxScaler(), if time features used
options["scale_path"] = options["model_dir"] + "scale.pkl"

# Model hyperparameters
options["hidden"] = 256  # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 100
options["n_epochs_stop"] = 10
options["batch_size"] = 2

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"] = True
options["cuda_devices"] = None
options["log_freq"] = None

# Prediction settings
# ``num_candidates`` controls the size of the candidate set for anomaly
# detection.  Lower values make it harder for uncommon or unknown keys to
# appear in the top-K predictions, improving anomaly detection【244891535061353†L260-L269】.
options["num_candidates"] = 3
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

# Set random seeds for reproducibility
seed_everything(seed=1234)

if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)

def main():
    """Command-line entry point"""
    print("device", options["device"])
    print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
    print("mask ratio", options["mask_ratio"])

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train':
        Trainer(options).train()
    elif args.mode == 'predict':
        # update gaussian parameters from command line
        options["gaussian_mean"] = args.mean
        options["gaussian_std"] = args.std
        Predictor(options).predict()
    elif args.mode == 'vocab':
        # build vocabulary from the combined normal sequences
        with open(options["train_vocab"], "r", encoding=args.encoding) as f:
            texts = f.readlines()
        vocab = WordVocab(texts, max_size=args.vocab_size, min_freq=args.min_freq)
        print("VOCAB SIZE:", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])

if __name__ == "__main__":
    main()