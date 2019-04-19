from __future__ import division
from __future__ import print_function

import os
import random


class Sample:
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath

# class Batch:
#     def __init__(self, gtTexts, imgs):
#         self.imgs = np.stack(imgs, axis=0)
#         self.gtTexts = gtTexts

class DataLoader:
    unique_character_file = '../model/unique_characters.txt'
    train_validation_words_file = '../data/corpus.txt'
    img_size = (128, 32)
    max_label_cost = 32
    
    def __init__(self, filePath, batch_size, data_size, train_data_per, augment_data=False):
        assert filePath[-1] == '/'
        self.augment_data = augment_data
        self.currIdx = 0
        self.batch_size = batch_size
        print(data_size)
        self.data_size = data_size
        self.samples = []
        f = open(filePath + 'words.txt')
        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
        
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue
            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9
            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

            # GT text are columns starting at 9
            gtText = self.truncateLabel(' '.join(lineSplit[8:]))
            chars = chars.union(set(list(gtText)))

            # check if image is not empty
            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            # put sample into list
            self.samples.append(Sample(gtText, fileName))

        # testing with small data set self.samples = self.samples[:500]
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        # split into training and validation set: train_data_per% - (1-train_data_per)%
        self.test_data_size = None
        if self.data_size < 100:
            self.test_data_size = int(len(self.samples)* (100-self.data_size)/100)
        self.data_size = int(len(self.samples) * self.data_size/100)
        #test_data_size = len(self.samples) - self.data_size
        splitIdx = int(train_data_per/100 * self.data_size)
        self.old_sample_set = self.samples.copy()
        ####This is for future to find generaization error with other modules
        if self.test_data_size:
            self.random_test_data = [random.randint(0, self.data_size) for _ in range(self.test_data_size)]
            f = open("./V7_50_test.txt", 'a')
            f.write(str(self.random_test_data))
            f.close()
            self.samples = [self.samples[i] for i in range(len(self.samples)) if i not in self.random_test_data]

        self.train_samples = self.samples[:splitIdx]
        self.validation_samples = self.samples[splitIdx:self.data_size]
        print(len(self.train_samples), len(self.validation_samples), self.test_data_size, "==", len(self.old_sample_set))
        self.train_words = [x.gtText for x in self.train_samples]
        self.validation_words = [x.gtText for x in self.validation_samples]
        self.char_list = sorted(list(chars))

        open(DataLoader.unique_character_file, 'w').write(str().join(self.char_list))
        open(DataLoader.train_validation_words_file, 'w').write(str(' ').join(self.train_words + self.validation_words))
    
    def truncateLabel(self, text):
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > self.max_label_cost:
                return text[:i]
        return text
    
    def get_random_train_set(self, samples_per_epoch):
        random.shuffle(self.train_samples)
        self.samples = self.train_samples[:samples_per_epoch]
        return self.samples
    
    def get_validation_set(self):
        return self.validation_samples

    def get_test_set(self):
        if self.test_data_size:
            return [self.old_sample_set[i] for i in self.random_test_data]
