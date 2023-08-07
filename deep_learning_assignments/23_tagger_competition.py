#!/usr/bin/env python3

from morpho_dataset import MorphoDataset
from morpho_analyzer import MorphoAnalyzer
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re
from typing import Any, Dict
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=64, type=int,
                    help="CLE embedding dimension.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int,
                    help="Maximum number of sentences to load.")
parser.add_argument("--rnn_cell", default="LSTM",
                    type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=128,
                    type=int, help="RNN cell dimension.")
parser.add_argument("--we_dim", default=64, type=int,
                    help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.2, type=float,
                    help="Mask words with the given probability.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")


class Model(tf.keras.Model):
    # A layer setting given rate of elements to zero.
    class MaskElements(tf.keras.layers.Layer):
        def __init__(self, rate: float) -> None:
            super().__init__()
            self._rate = rate

        def get_config(self) -> Dict[str, Any]:
            return {"rate": self._rate}

        def call(self, inputs: tf.RaggedTensor, training: bool) -> tf.RaggedTensor:
            if training:
                mask = tf.random.uniform(shape=tf.shape(inputs.values))
                zeros = tf.zeros_like(inputs.values)
                masked = tf.where(tf.greater(
                    mask, self._rate), inputs.values, zeros)
                return inputs.with_values(masked)

            else:
                return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        words = tf.keras.layers.Input(
            shape=[None], dtype=tf.string, ragged=True)
        indicesBasic = train.forms.word_mapping(words)
        indices = self.MaskElements(rate=args.word_masking)(indicesBasic)
        embedded_words = tf.keras.layers.Embedding(
            input_dim=train.forms.word_mapping.vocabulary_size(),
            output_dim=args.we_dim)(indices)

        # Create a vector of input words from all batches using `words.values`
        # and pass it through `tf.unique`, obtaining a list of unique words and
        # indices of the original flattened words in the unique word list.
        unique_words, unique_indices = tf.unique(words.values)

        # Create sequences of letters by passing the unique words through
        # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.
        letters = tf.strings.unicode_split(
            input=unique_words,
            input_encoding="UTF-8")

        # Map the letters into ids by using `char_mapping` of `train.forms`.
        letter_ids = train.forms.char_mapping(letters)

        # Embed the input characters with dimensionality `args.cle_dim`.
        embedded_letters = tf.keras.layers.Embedding(
            input_dim=train.forms.char_mapping.vocabulary_size(),
            output_dim=args.cle_dim
        )(letter_ids)

        # Pass the embedded letters through a bidirectional GRU layer
        # with dimensionality `args.cle_dim`, obtaining character-level representations
        # of the whole words, **concatenating** the outputs of the forward and backward RNNs.
        rnn_letters = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(args.cle_dim),
            merge_mode="concat"
        )(embedded_letters)

        # Use `tf.gather` with the indices generated by `tf.unique` to transform
        # the computed character-level representations of the unique words to representations
        # of the flattened (non-unique) words.
        # Todo: isn't this the same as rnn_letters?
        cle = tf.gather(rnn_letters, unique_indices)

        # Then, convert these character-level word representations into
        # a RaggedTensor of the same shape as `words` using `words.with_values` call.
        cle_correct_shape = embedded_words.with_values(cle)

        # Concatenate the word-level embeddings and the computed character-level WEs
        # (in this order).
        concatenated = tf.keras.layers.Concatenate()(
            [embedded_words, cle_correct_shape])

        # TODO: Add a dropout layer with rate `args.dropout`.

        # (tagger_we): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the word representations, **summing** the outputs of forward and backward RNNs.
        if args.rnn_cell == "LSTM":
            rnn_cell = tf.keras.layers.LSTM(
                units=args.rnn_cell_dim,
                return_sequences=True
            )
        elif args.rnn_cell == "GRU":
            rnn_cell = tf.keras.layers.GRU(
                units=args.rnn_cell_dim,
                return_sequences=True
            )

        layer = tf.keras.layers.Bidirectional(
            rnn_cell,
            merge_mode='sum')(concatenated)

        # (tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a RaggedTensor without any problem.
        last_layer = tf.keras.layers.Dense(
            units=train.tags.word_mapping.vocabulary_size(),
            activation='softmax')(layer)

        # TODO: Plugin the correct thing
        predictions = last_layer

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)
        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=lambda yt, yp: tf.losses.SparseCategoricalCrossentropy()(yt.values, yp.values),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(
            re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the model and train
    model = Model(args, morpho.train)

    # (tagger_we): Construct dataset for training, which should contain pairs of
    # - tensor of string words (forms) as input
    # - tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(example):
        data = example["forms"]
        targets = morpho.train.tags.word_mapping(example["tags"])
        return data, targets

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(tagging_dataset)
        dataset = dataset.shuffle(
            len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev, test = create_dataset(
        "train"), create_dataset("dev"), create_dataset("test")

    logs = model.fit(train, epochs=args.epochs,
                     validation_data=dev, callbacks=[model.tb_callback])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set; update the following prediction
        # command if you use other output structre than in tagger_we.
        predictions = model.predict(test)
        tag_strings = morpho.test.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in sentence:
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
