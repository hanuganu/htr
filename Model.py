import os
import numpy as np
import tensorflow as tf

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2

class Model:
	batch_size = 50
	input_img_size = (128, 32)
	max_text_len = 32
	version = "T_V8"
	trained_modules = '../model/trained_models/'
	optimizers = {
		0:tf.train.RMSPropOptimizer,
		1:tf.train.AdamOptimizer,
		2:tf.train.AdagradDAOptimizer,
		3:tf.train.AdagradOptimizer
	}
	def __init__(self, char_list, optimizer = 0, re_used=False):
		self.char_list = char_list
		self.decoder_type = DecoderType.BestPath
		self.re_train = re_used
		self.version_ID = 0
		if(optimizer>3):
			optimizer = 0
		optimizer = Model.optimizers[optimizer]
		print("Optimizer used:",optimizer.__name__)
		self.is_train = tf.placeholder(tf.bool, name='is_train')
		self.input_imgs = tf.placeholder(tf.float32, shape=(None, Model.input_img_size[0], Model.input_img_size[1]))
		self.setupCNN()
		self.setupRNN()
		self.setupCTC()
		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
		with tf.control_dependencies(self.update_ops):
			self.optimizer = optimizer(self.learning_rate).minimize(self.loss)
		(self.sess, self.saver) = self.setupTF()
			
			
	def setupCNN(self):
		"create CNN layers and return output of these layers"
		cnnIn4d = tf.expand_dims(input=self.input_imgs, axis=3)
		kernelVals = [3, 3, 3, 3, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128,128 ,128, 256]
		strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2), (1,1), (1,1)]
		numLayers = len(strideVals)
		pool = cnnIn4d  # input to first CNN layer
		for i in range(numLayers):
					kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
					conv = tf.nn.conv2d(pool, kernel, padding='SAME', 	strides=(1, 1, 1, 1))
					conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
					relu = tf.nn.relu(conv_norm)
					pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
					self.cnnOut4d = pool

	def setupRNN(self):
			"create RNN layers and return output of these layers"
			rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])
			numHidden = 256
			cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)]  # 2 layers
			stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
			((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
			concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
			kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.char_list) + 1], stddev=0.1))
			self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

	def setupCTC(self):
			self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
			self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
			self.seqLen = tf.placeholder(tf.int32, [None])
			self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))
			self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.max_text_len, None, len(self.char_list) + 1])
			self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)
			if self.decoder_type == DecoderType.BestPath:
				self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)

	def setupTF(self):
		sess = tf.Session()  # TF session
		saver = tf.train.Saver(max_to_keep=1)  # saver saves model to file
		if self.re_train:
			saved_model = self.is_saved_model()
			if saved_model:
				print('Init with stored values from ' + saved_model)
				saver.restore(sess, saved_model)
			else:
				raise Exception('No saved model found in: ' + self.new_model)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())
		return (sess, saver)
	
	def is_saved_model(self):
		return tf.train.latest_checkpoint(self.new_model)

	def toSparse(self, texts):
			indices = []
			values = []
			shape = [len(texts), 0]
			for (batchElement, text) in enumerate(texts):
				labelStr = [self.char_list.index(c) for c in text]
				if len(labelStr) > shape[1]:
					shape[1] = len(labelStr)
				for (i, label) in enumerate(labelStr):
					indices.append([batchElement, i])
					values.append(label)
			return (indices, values, shape)

	def decoderOutputToText(self, ctcOutput, batchSize):
			encodedLabelStrs = [[] for i in range(batchSize)]
			decoded = ctcOutput[0][0]
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0]
				encodedLabelStrs[batchElement].append(label)
			return [str().join([self.char_list[c] for c in labelStr]) for labelStr in encodedLabelStrs]

	def train_batch(self, texts, images, learning_rate):
			"feed a batch into the NN to train it"
			numBatchElements = len(images)
			sparse = self.toSparse(texts)
			evalList = [self.optimizer, self.loss]
			feedDict = {self.input_imgs : images, self.gtTexts : sparse , self.seqLen : [Model.max_text_len] * numBatchElements, self.learning_rate : learning_rate, self.is_train: True}
			(_, lossVal) = self.sess.run(evalList, feedDict)
			return lossVal

	def inferBatch(self, image_text, images, calcProbability=False, probabilityOfGT=False):
			numBatchElements = len(images)
			evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
			feedDict = {self.input_imgs : images, self.seqLen : [Model.max_text_len] * numBatchElements, self.is_train: False}
			evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
			decoded = evalRes[0]
			texts = self.decoderOutputToText(decoded, numBatchElements)
			probs = None
			if calcProbability:
				sparse = self.toSparse(image_text) if probabilityOfGT else self.toSparse(texts)
				ctcInput = evalRes[1]
				evalList = self.lossPerElement
				feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.max_text_len] * numBatchElements, self.is_train: False}
				lossVals = self.sess.run(evalList, feedDict)
				probs = np.exp(-lossVals)
			return (texts, probs)

	def save(self, model_config):
		self.version_ID += 1
		current_model = Model.trained_modules + model_config
		try:
			os.mkdir(current_model)
		except:
			print(current_model, " already exist")
		self.saver.save(self.sess, current_model + "/version", global_step=self.version_ID)
 
