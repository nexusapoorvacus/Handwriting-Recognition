from PIL import Image
import tensorflow as tf
import tflib
import tflib.ops
import tflib.network
from tqdm import tqdm
import numpy as np
import data_loaders
import time
import os
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', action='store', default=0.01)
parser.add_argument('--decay_rate', action='store', default=2)
parser.add_argument('--num_epochs', action='store', default=1)
parser.add_argument('--num_iterations', action='store', default=1000)
parser.add_argument('--optimizer', action='store', default="sgd")
parser.add_argument('--batch_size', action='store', default='16')
parser.add_argument('--embedding_size', action='store', default=60)

args = parser.parse_args()
print(args)

lr = float(args.lr)
decay_rate = int(args.decay_rate)
num_epochs = int(args.num_epochs)
num_iterations = int(args.num_iterations)
optimizer_type = args.optimizer
emb_size = int(args.embedding_size)

BUCKETS_DIR = "/data/apoorvad/cs221proj/words/"
PROPERTIES_DIR = "/data/apoorvad/cs221proj/words/properties.npy"
PROCESSED_IMAGES_DIR = "/data/apoorvad/cs221proj/words/processed/"
WEIGHTS_CHECKPOINT_FILE = "/data/apoorvad/cs221proj/words/checkpoints/weights_best.ckpt"
SAVE_ATT_IMGS = "/data/apoorvad/cs221proj/words/att_imgs/"

BATCH_SIZE      = int(args.batch_size)
EMB_DIM         = emb_size
ENC_DIM         = 256
DEC_DIM         = ENC_DIM*2
NUM_FEATS_START = 64
D               = NUM_FEATS_START*8
V               = 502
NB_EPOCHS       = num_epochs
H               = 20
W               = 50

X = tf.placeholder(shape=(None,None,None,None),dtype=tf.float32)
mask = tf.placeholder(shape=(None,None),dtype=tf.int32)
seqs = tf.placeholder(shape=(None,None),dtype=tf.int32)
learn_rate = tf.placeholder(tf.float32)
input_seqs = seqs[:,:-1]
target_seqs = seqs[:,1:]
emb_seqs = tflib.ops.Embedding('Embedding',V,EMB_DIM,input_seqs)

ctx = tflib.network.im2latex_cnn(X,NUM_FEATS_START,True)
out,state = tflib.ops.im2latexAttention('AttLSTM',emb_seqs,ctx,EMB_DIM,ENC_DIM,DEC_DIM,D,H,W)
logits = tflib.ops.Linear('MLP.1',out,DEC_DIM,V)
predictions = tf.argmax(tf.nn.softmax(logits[:,-1]),axis=1)


loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=tf.reshape(logits,[-1,V]),
    labels=tf.reshape(seqs[:,1:],[-1])
    ), [tf.shape(X)[0], -1])

mask_mult = tf.to_float(mask[:,1:])
loss = tf.reduce_sum(loss*mask_mult)/tf.reduce_sum(mask_mult)

if optimizer_type == "adam":
  train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
elif optimizer_type == "sgd": # argv[2] == "SGD"
  optimizer = tf.train.GradientDescentOptimizer(learn_rate)
  gvs = optimizer.compute_gradients(loss)
  capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
  train_step = optimizer.apply_gradients(capped_gvs)
else:
  train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(loss)

def predict(set='valid',batch_size=BATCH_SIZE,visualize=False):
    if visualize:
        assert (batch_size==1), "Batch size should be 1 for visualize mode"
    import random
    # f = np.load('train_list_buckets.npy').tolist()
    files = np.load(BUCKETS_DIR+set+'_buckets.npy').tolist()
    #random_key = random.choice(f.keys())
    #random_key = (160,40)
    
    imgs = []
    true_labels = []
    #print "Image shape: ",random_key
    for bucket in files:
            if bucket in [(240, 40)]:
              continue
            print "bucket", bucket
            f = files[bucket]
	    while len(imgs)!=batch_size and len(imgs) != len(f):
		start = np.random.randint(0,len(f),1)[0]
		if os.path.exists(PROCESSED_IMAGES_DIR+f[start][0]):
		    true_labels.append(f[start][1])
		    imgs.append(np.asarray(Image.open(PROCESSED_IMAGES_DIR+f[start][0]).convert('YCbCr'))[:,:,0][:,:,None])

	    imgs = np.asarray(imgs,dtype=np.float32).transpose(0,3,1,2)
	    inp_seqs = np.zeros((batch_size,50)).astype('int32')
	    print imgs.shape
	    inp_seqs[:,0] = np.load(PROPERTIES_DIR).tolist()['char_to_idx']['#START']
	    tflib.ops.ctx_vector = []

	    l_size = bucket[0]*2
	    r_size = bucket[1]*2
	    inp_image = Image.fromarray(imgs[0][0]).resize((l_size,r_size))
	    l = int(np.ceil(bucket[1]/8.))
	    r = int(np.ceil(bucket[0]/8.))
	    properties = np.load(PROPERTIES_DIR).tolist()
	    idx_to_chars = lambda Y: ' '.join(map(lambda x: properties['idx_to_char'][x],Y))

	    for i in xrange(1,50):
                try:
		  inp_seqs[:,i] = sess.run(predictions,feed_dict={X:imgs,input_seqs:inp_seqs[:,:i]})
                except:
                  continue
		#print i,inp_seqs[:,i]
		if visualize==True:
		    att = sorted(list(enumerate(tflib.ops.ctx_vector[-1].flatten())),key=lambda tup:tup[1],reverse=True)
		    idxs,att = zip(*att)
		    j=1
		    while sum(att[:j])<0.9:
			j+=1
		    positions = idxs[:j]
		    print "Attention weights: ",att[:j]
		    positions = [(pos/r,pos%r) for pos in positions]
		    outarray = np.ones((l,r))*255.
		    for loc in positions:
			outarray[loc] = 0.
		    out_image = Image.fromarray(outarray).resize((l_size,r_size),Image.NEAREST)
		    print "Latex sequence: ",idx_to_chars(inp_seqs[0,:i])
		    outp = Image.blend(inp_image.convert('RGBA'),out_image.convert('RGBA'),0.5)
		    outp.save(SAVE_ATT_IMGS + "image_att" + str(i) + ".png", title=properties['idx_to_char'][inp_seqs[0,i]])
    acc = find_accuracy(true_labels, inp_seqs)
    print("accuracy: " + str(acc))
    np.save('/data/apoorvad/cs221proj/words/pred_imgs',imgs)
    np.save('/data/apoorvad/cs221proj/words/pred_latex',inp_seqs)
    print "Saved npy files! Use Predict.ipynb to view results"
    return acc
    #return inp_seqs

def find_accuracy(true_chars, pred_chars):
  print("true", true_chars)
  print("pred", pred_chars)
  good = 0
  total = 0
  for i in range(len(true_chars)):
    for c in range(len(true_chars[i])):
      if true_chars[i][c] == pred_chars[i][c]:
        good += 1
      total += 1
  return (good + 0.0) / total
   

def score(set='valid',batch_size=BATCH_SIZE):
    score_itr = data_loaders.data_iterator(set,batch_size)
    losses = []
    start = time.time()
    for score_imgs,score_seqs,score_mask in score_itr:
        _loss = sess.run(loss,feed_dict={X:score_imgs,seqs:score_seqs,mask:score_mask})
        losses.append(_loss)

    set_loss = np.mean(losses)
    perp = np.mean(map(lambda x: np.power(np.e,x), losses))
    print "\tMean %s Loss: ", set_loss
    print "\tTotal %s Time: ", time.time()-start
    print "\tMean %s Perplexity: ", perp
    sys.stdout.flush()
    return set_loss, perp

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
lr = float(args.lr)
#if os.path.exists(WEIGHTS_CHECKPOINT_FILE + "_0.01.meta"):
#  saver = tf.train.import_meta_graph(WEIGHTS_CHECKPOINT_FILE + "_0.01.meta")
#  saver.restore(sess, WEIGHTS_CHECKPOINT_FILE+"_0.01")
#else:
sess.run(init)

def train():
  losses = []
  times = []
  print "Compiled Train function!"
  sys.stdout.flush()
  i=0
  best_perp = np.finfo(np.float32).max
  best_loss = np.finfo(np.float32).max
  
  for i in xrange(i,NB_EPOCHS):
    iter=0
    costs=[]
    times=[]
    itr = data_loaders.data_iterator('train', BATCH_SIZE)
    for train_img,train_seq,train_mask in itr:
        if i == (NB_EPOCHS - 1) and iter >= num_iterations:
          break
        iter += 1
        start = time.time()
        _ , _loss = sess.run([train_step,loss],feed_dict={X:train_img,seqs:train_seq,mask:train_mask,learn_rate:float(args.lr)})

        times.append(time.time()-start)
        costs.append(_loss)
        #if iter%100==0:
        print "Iter: %d (Epoch %d)"%(iter,i+1)
        print "\tMean cost: ",np.mean(costs)
        print "\tMean time: ",np.mean(times)
        sys.stdout.flush()

    print "\n\nEpoch %d Completed!"%(i+1)
    print "\tMean train cost: ",np.mean(costs)
    print "\tMean train perplexity: ",np.mean(map(lambda x: np.power(np.e,x), costs))
    print "\tMean time: ",np.mean(times)
    sys.stdout.flush()
    val_loss, val_perp = score('valid',BATCH_SIZE)
    if val_loss < best_loss:
        best_loss = val_loss
        saver.save(sess,WEIGHTS_CHECKPOINT_FILE)
        print "\tBest Validation Loss Till Now! Saving state!"
    else:
        lr = lr * (1.0/decay_rate)
    print "Epoch Accuracy: " + str(predict())
    sys.stdout.flush()

train()
