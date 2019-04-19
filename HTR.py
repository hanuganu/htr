'''
Created on २६ जाने, २०१९

@author: JH-ANIC
'''
import sys
try:
	import  cv2
except ImportError:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import json
from SamplePreprocessor import preprocess
from DataLoader import DataLoader
from Model import Model, DecoderType
import editdistance
from getopt import getopt, GetoptError
import numpy

class HandWrittenTextRecognizer(object):

    train_data_files = '../data/'
    infer_image_file = '../data/test.png'
    reporting_files = '../data/results/'
    model_config = ""
    def __init__(self, argv):
	
        a_options = "vardit:e:l:s:b:p:x:y:o:"
        self.optimizer_code = 0
        self.limit_early_stop = 0
        self.learning_rate = 0.01
        self.epochs = 0
        self.samples_per_epoch = 5000
        self.infer = False
        self.train_data_per = 30
        self.validate = False
        self.re_used = False #improve trained model
        self.train = False
        self.batch_size = 1 #default 1
        self.sample_size_limit = 0 #default tet
        self.decay_rate = 0 #decay rate
        self.augment = False
        self.is_early_stop = False
        try:
            opts, args = getopt(argv,a_options,["train=","learning_rate=","limit_early_stop=","batch_size=", "test_data_size="])
        except GetoptError:
            print("Command line option\n Usage: htr.py [-t <epoch>] [-l <learning rate>] [-e <condition>] [-v -i -a -r]")
            sys.exit(1)
        for option, argument in opts:
            if option == '-l':
                self.learning_rate = float(argument)
            elif option == '-b':
                self.batch_size = int(argument)
            elif option == '-p':
                self.samples_per_epoch = int(argument)
            elif option == '-x':
                self.train_data_per = float(argument)
            elif option == '-s':
                self.sample_size_limit = 100 if float(argument) > 100 or float(argument) < 0 else float(argument)
            elif option == '-d':
                self.decay_rate = 1
            elif option == '-v':
                self.validate = True
            elif option == '-i':
                self.infer = True
            elif option == '-a':
                self.augment = True
            elif option == '-r':
                self.re_used = True
            elif option == '-o':
                self.optimizer_code = 0 if int(argument) < 1 else int(argument)

            if option in ['-t','-e']:
                if option == '-t':
                    self.epochs = int(argument)
                elif option == '-e':
                    self.limit_early_stop = int(argument)
                    self.is_early_stop = True
                if(not self.train):
                    self.train = True
                else:
                    print("Error in command line options:\n options -t/--train and -e/--limit_early_stop are mutually exclusive")
                    sys.exit(2)
        loader = DataLoader(HandWrittenTextRecognizer.train_data_files, self.batch_size, self.sample_size_limit, self.train_data_per)
        self.__define_module()
        test_set = None
        if self.sample_size_limit <100:
            test_set = loader.get_test_set()
            print(len(test_set),"Is it Correct?")

        if self.train:
            model = Model(loader.char_list, self.optimizer_code, self.re_used)
            self.__train_model(model, loader.get_random_train_set, loader.get_validation_set(), test_set)


        elif self.validate:
            model = Model(loader.char_list,self.optimizer_code, re_used=True)
            self.__validate_model(model, loader.get_validation_set())
    
        else:
            print(open(Model.trained_modules + HandWrittenTextRecognizer.model_config).read())
            model = Model(open(DataLoader.unique_character_file).read(), self.optimizer_code, re_used=True)
            self.infer(model, HandWrittenTextRecognizer.infer_image_file)

    def __batch_generator(self, samples, augmentation):
        no_of_batches = 0
        if augmentation == None:
            batch_size = self.validation_batch_size
            augmentation = False
        else:
            batch_size = self.batch_size
        batches_possible = len(samples)//batch_size
        while no_of_batches < batches_possible:
            img_txt = []
            images = []
            for index in range(no_of_batches*batch_size,(no_of_batches+1)*batch_size):
                img_txt.append(samples[index].gtText)
                images.append(preprocess(cv2.imread(samples[index].filePath, cv2.IMREAD_GRAYSCALE), Model.input_img_size, augmentation))
            yield (img_txt.copy(), numpy.stack(images.copy(),axis=0))
            no_of_batches += 1
            
    def __verify_batch_samples_per_epoch(self, samples):
        reset_config = False
        if (self.batch_size > len(samples)):
            self.batch_size = len(self.samples)
            print("Warning: Batch size adjusted to ", self.batch_size)
            reset_config = True
        if (len(samples)< self.samples_per_epoch):
            self.samples_per_epoch = len(samples)
            reset_config = True
        if reset_config:
            self.__define_module()

    def __define_module(self):
        HandWrittenTextRecognizer.model_config = Model.version +"_OPC_"+str(self.optimizer_code) +"_TE_" + str(self.epochs) + "_ES_" + str(
            self.limit_early_stop) + "_SPE_" + str(self.samples_per_epoch) + "_B_" + str(
            self.batch_size) + "_LR_" + str(self.learning_rate) + "_DLR_" + str(self.decay_rate) + "_TDP_" + str(
            self.train_data_per)

    def __train_model(self, model, get_random_train_set, validation_samples, test_set = None):
        "compile NN"
        epoch = 0
        min_error_rate = float('inf')
        early_stop_counter = 0
        counter = 0
        training_samples = get_random_train_set(self.samples_per_epoch)
        self.__verify_batch_samples_per_epoch(training_samples)
        print("Model configuration:",HandWrittenTextRecognizer.model_config)
        if (len(validation_samples) < self.batch_size):
            self.validation_batch_size = len(validation_samples)
        else:
            self.validation_batch_size = self.batch_size
        while True:
            epoch += 1
            my_batch_generator = self.__batch_generator(training_samples, self.augment)
            obj = []
            obj.append(str(epoch))
            obj.append(str(self.learning_rate))
            for batch_number, txt_img_batch in enumerate(my_batch_generator):
                learning_rate = self.learning_rate 
                if self.decay_rate > 0:
                    learning_rate = learning_rate if counter < 10 else (learning_rate/10 if counter< 10000 else learning_rate/100)
                counter += 1
                loss = model.train_batch(txt_img_batch[0],txt_img_batch[1],learning_rate)
                obj.append(str(loss))
                print('Batch:', batch_number+1,'/', len(training_samples)//self.batch_size, 'Loss:', loss)

            error_rate, word_error_rate = self.__validate_model(model, validation_samples)
            obj.append(str(error_rate))
            obj.append(str(word_error_rate))

            self.__reporting(HandWrittenTextRecognizer.reporting_files+HandWrittenTextRecognizer.model_config+".txt", obj)

            if error_rate < min_error_rate:
                print('Saving model..')
                min_error_rate = error_rate
                early_stop_counter = 0
                model.save(HandWrittenTextRecognizer.model_config)
                open(Model.trained_modules + HandWrittenTextRecognizer.model_config +"/accuracy.txt", 'w').write('Validation character error rate of saved model: %f%%' % (error_rate*100.0))
            
            elif self.is_early_stop:
                early_stop_counter += 1
                if early_stop_counter >= self.limit_early_stop:
                    print('Limit reached for early stop %d' %(self.limit_early_stop))
                    break
                else:
                    print('No improvement, Counter becomes',early_stop_counter)
            
            elif self.epochs == epoch:
                print('Limit reached for epoch stop %d' %(self.epochs))
                break

            training_samples = get_random_train_set(self.samples_per_epoch)

        if test_set:
            obj = []
            if(len(test_set)<self.validation_batch_size):
                self.validation_batch_size = len(test_set)
            charErrorRate, wordAccuracy = self.__validate_model(model, test_set)
            print("+======================================================================+")
            print(charErrorRate, wordAccuracy)
            obj.append("Character Error Rate:"+str(charErrorRate))
            obj.append("Word accuracy:"+str(wordAccuracy))
            self.__reporting(HandWrittenTextRecognizer.reporting_files + HandWrittenTextRecognizer.model_config + "_test_error_.txt", obj)

    def __reporting(self, file_name, msg):
        with open(file_name, 'a') as reporting_file:
            reporting_file.write("\n"+",".join(msg))
            reporting_file.close()

    def __validate_model(self, model, validation_samples):
        wrong_characters = 0
        total_charaters = 0
        correct_words = 0
        total_words = 0
        for batch_number, txt_img_batch in enumerate(self.__batch_generator(validation_samples,None)):
            print('Validation Batch:', batch_number+1,'/', len(validation_samples)//self.validation_batch_size +1)
            (recognized, _) = model.inferBatch(txt_img_batch[0], txt_img_batch[1])
            for i in range(len(recognized)):
                correct_words += 1 if txt_img_batch[0][i] == recognized[i] else 0
                total_words += 1
                dist = editdistance.eval(recognized[i], txt_img_batch[0][i])
                wrong_characters += dist
                total_charaters += len(txt_img_batch[0][i])
        
        charErrorRate = wrong_characters / total_charaters
        wordAccuracy = correct_words / total_words
        print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
        return charErrorRate, wordAccuracy


    # def infer(self, model, fnImg):
    #     "recognize text in image provided by file path"
    #     img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.input_image_size)
    #     batch = Batch(None, [img])
    #     (recognized, probability) = model.inferBatch(batch, True)
    #     print('Recognized:', '"' + recognized[0] + '"')
    #     print('Probability:', probability[0])

if (__name__ =="__main__"):    
    htr_system = HandWrittenTextRecognizer(sys.argv[1:])
    
