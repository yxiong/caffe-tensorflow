#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Jun 17, 2016.

"""Export model to frozen protobuf format, so that they can be loaded on client or elsewhere."""

import argparse
import numpy as np
import os.path
import shutil
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import tempfile

import models

def export(npy_path, pb_path):
    print "Creating the network ..."
    spec = models.get_data_spec(model_class=models.GoogleNet)
    input_node = tf.placeholder(
        tf.float32,
        name = "input",
        shape=(None, spec.crop_size, spec.crop_size, spec.channels))
    net = models.GoogleNet({'data': input_node})

    with tf.Session() as sess:
        print "Loading numpy data into model ..."
        net.load(npy_path, sess)

        tempd = tempfile.mkdtemp()
        print "Creating folder for temporary files", tempd, "..."

        print "Writing parameters to a checkpoint ..."
        checkpoint_file = os.path.join(tempd, "checkpoint")
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess, checkpoint_file, global_step=0)

        print "Writing model (with no parameters) to a file ..."
        graph_file = os.path.join(tempd, "graph_with_no_params.pb")
        tf.train.write_graph(sess.graph_def, "", graph_file, as_text=False)

        print "Freezing parameters into model as a protobuf file ..."
        freeze_graph.freeze_graph(
            input_graph = graph_file,
            input_saver = "",
            input_binary = True,
            input_checkpoint = checkpoint_file + "-0",
            output_node_names = "prob",
            restore_op_name = "save/restore_all",
            filename_tensor_name = "save/Const:0",
            output_graph = pb_path,
            clear_devices = False,
            initializer_nodes = "")

        print "Removing temporary directory ..."
        shutil.rmtree(tempd)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_path', help='Converted parameters for the GoogleNet model.')
    parser.add_argument('pb_path', help='Protobuf file to be exported to.')
    args = parser.parse_args()

    # Export the model
    export(args.npy_path, args.pb_path)

if __name__ == "__main__":
    main()
