import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf2_1
import tf2_2
TEST_INTERVAL_SECS=10

def test(mnist):
	x=tf.placeholder(tf.float32,[None,tf2_1.INPUT_NODE])
	y_=tf.placeholder(tf.float32,[None,tf2_1.OUTPUT_NODE])
	y=tf2_1.forward(x,None)
	
	ema=tf.train.ExponentialMovingAverage(tf2_2.MOVING_AVERAGE_DECAY)
	ema_restore=ema.variables_to_restore()
	saver=tf.train.Saver(ema_restore)
	
	correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	
	while True:
		with tf.Session() as sess:
			ckpt=tf.train.get_checkpoint_state(tf2_2.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
				global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				accuracy_score=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
				print('After %s training steps, test accuracy= %g'%(global_step,accuracy_score))
			else:
				print('No checkpoint file found')
				return
		time.sleep(TEST_INTERVAL_SECS)
		
def main():
	mnist=input_data.read_data_sets("C:/Git/TF/tf2/data/",one_hot=True)
	test(mnist)
	
if __name__=='__main__':
	main()