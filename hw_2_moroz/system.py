import tensorflow as tf
import json
import numpy as np
import argparse
import logging

# parsing argument with a name of a file containing parameters of a systemof equations
def parse_args():
    parser = argparse.ArgumentParser(description="Input for a filename with params for a system of equations")
    parser.add_argument('--path',type=str,required=True,help='Filename with params for a systemof equations')
    return parser.parse_args()

#configuring logging
logging.basicConfig(level=logging.DEBUG,filename="app_system.log",filemode="w")

logging.info("Start program...")

path = parse_args().path

#deserialisimg parameters from json-file
try:
    with open(path, "r") as json_file:
        params = json.load(json_file)
except:
    logging.error("No file with params or invalid params!")
else:
    #finding roots of the system of equation by inverse matrix method
    try:
        m_coeff = tf.constant(np.array(params["coefficients"],dtype=float))
        m_const = tf.constant(np.array(params["constants"],dtype=float).T)
        m_roots = tf.matmul(tf.linalg.inv(m_coeff),m_const)
        m_invalid_roots = tf.matmul(tf.transpose(m_const),tf.linalg.inv(m_coeff),)

        with tf.Session() as sess:
            logging.info("Vector of solutions is:")
            logging.info(sess.run(m_roots))
            logging.info("Vector of incorrect solutions, obtained through changing the order of the matrix multipliers,is:")
            logging.info(sess.run(m_invalid_roots))
            logging.info("Are vectors equal?")
            logging.info(sess.run(tf.equal(m_invalid_roots,m_roots)))
    except:
        logging.error("Exception during calculation occurred!", exc_info=True)










# tensor

#x = tf.range(12)
#with tf.Session() as sess: print(sess.run(x))
