import tensorflow as tf
import argparse
import logging
import json

# parsing argument with a name of a file containing parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Input for a filename with params")
    parser.add_argument('--path',type=str,required=True,help='Filename with params')
    return parser.parse_args()

#configuring logging
logging.basicConfig(level=logging.DEBUG,filename="app_activation.log",filemode="w")
logging.info("Start program...")

path = parse_args().path

try:
    with open(path, "r") as json_file:
        params = json.load(json_file)
except:
    logging.error("No file with params or invalid params!")
else:
    try:
        #processing params
        x_scalar = tf.constant([params["scalar"]])
        x_vector = tf.constant(params["vector"])

        #calculating derivatives of activation functions

        #Logistics function

        def f_logist(x):
            x = tf.Variable(x)
            with tf.GradientTape(persistent=True) as t:
                y = 1 / (1 + tf.math.exp(-x))
                x_grad = t.gradient(y, x)
            return x_grad

        logging.info("Scalar derivative from Logistics function at point {}:".format(params["scalar"]))
        logging.info(f_logist(x_scalar))

        logging.info("Vector derivative from Logistics function for vector {}:".format(params["vector"]))
        logging.info(f_logist(x_vector))
        logging.info("-----------------------------------------------------------------")

        #SoftPlus function

        def f_softplus(x):
            x = tf.Variable(x)
            with tf.GradientTape(persistent=True) as t:
                y= tf.math.log(1+tf.math.exp(x))
                x_grad = t.gradient(y, x)
            return x_grad


        logging.info("Scalar derivative from SoftPlus function at point {}:".format(params["scalar"]))
        logging.info(f_softplus(x_scalar))

        logging.info("Vector derivative from SoftPlus function for vector {}:".format(params["vector"]))
        logging.info(f_softplus(x_vector))
        logging.info("-----------------------------------------------------------------")

        #ArcTan function

        def f_arctan(x):
            x = tf.Variable(x)
            with tf.GradientTape(persistent=True) as t:
                y= tf.math.atan(x)
                x_grad = t.gradient(y, x)
            return x_grad


        logging.info("Scalar derivative from ArcTan function at point {}:".format(params["scalar"]))
        logging.info(f_arctan(x_scalar))

        logging.info("Vector derivative from Arctan function for vector {}:".format(params["vector"]))
        logging.info(f_arctan(x_vector))

    except:
        logging.error("Exception during calculation occurred!", exc_info=True)





