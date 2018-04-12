import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tf1_1
import tf1_2

STEPS=40000
BATCH_SIZE=30
LEARNING_RATE_BASE=0.01
LEARNING_RATE_DECAY=0.999
REGULARIZER=0.01

def backward():
	x=tf.placeholder(tf.float32,shape=(None,2))
	y_=tf.placeholder(tf.float32,shape=(None,1)
	