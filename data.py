import csv
import random
import numpy as np

class DataSet:
    def __init__(self, name):
        self.name = name

        self.source_normal = []
        self.source_anomaly = []
        self.target_unlabeled = []
        self.target_test = []
        self.target_test_label = []

        self.read()
 

        # four_tuples (x, x-n, x+n, x!n) 
        self.four_tuples = []
        # (x!n, x-n, x)
        self.tuple_domain = []
        # (x-n, x, x+n)
        self.tuple_class = []
        self.train_label = []

        self.reorganize()

        self.train_input = []
        self.train_output = []
        self.test_input = []
        self.test_output = []

        self.fit_for_model()

    def read(self):
        input_file_normal = 'data/' + self.name + '/normal.csv'
        input_file_anomaly = 'data/' + self.name + '/anomaly.csv'
        input_file_unlabeled = 'data/' + self.name + '/unlabled.csv'
        input_file_test = 'data/' + self.name + '/test.csv'

        with open(input_file_normal,'r') as fin:
            lines = csv.reader(fin)
            for line in lines:
                tmp = [float(x) for x in line]
                self.source_normal.append(tmp)

        with open(input_file_anomaly,'r') as fin:
            lines = csv.reader(fin)
            for line in lines:
                tmp = [float(x) for x in line]
                self.source_anomaly.append(tmp)   

        with open(input_file_unlabeled,'r') as fin:
            lines = csv.reader(fin)
            for line in lines:
                tmp = [float(x) for x in line]
                self.target_unlabeled.append(tmp)

        with open(input_file_test,'r') as fin:
            lines = csv.reader(fin)
            for line in lines:
                tmp = [float(x) for x in line[:5]]
                self.target_test.append(tmp)
                self.target_test_label.append(int(line[-1]))

    """
    we have already done truncate when we clean the data first time
    """
    def truncate(self):
        pass

    """
    we have already done relabel when we clean the data first time
    """
    def relabel(self):
        pass

    """
    format four_tuples
    """
    def reorganize(self):
        random.shuffle(self.source_anomaly)
        random.shuffle(self.source_normal)
        random.shuffle(self.target_unlabeled)
        for i in self.source_anomaly[:20]:
            for j in self.source_anomaly[:20]:
                if i == j:
                    continue
                for k in self.source_normal[:20]:
                    for l in self.target_unlabeled[:20]:
                        self.four_tuples.append([i, j, k, l])
                        self.tuple_domain.append([l, j, i])
                        self.tuple_class.append([j, i, k])
                        self.train_label.append(1)

        for i in self.source_normal[:20]:
            for j in self.source_normal[:20]:
                if i == j:
                    continue
                for k in self.source_anomaly[:20]:
                    for l in self.target_unlabeled[:20]:
                        self.four_tuples.append([i, j, k, l])
                        self.tuple_domain.append([l, j, i])
                        self.tuple_class.append([j, i, k])
                        self.train_label.append(0)

    def fit_for_model(self):
        l = len(self.four_tuples)
        l0 = round(l * 0.8)
        self.train_input = [[],[],[],[]] 
        for i in self.four_tuples[:l0]:
            self.train_input[0].append(i[0])
            self.train_input[1].append(i[1])            
            self.train_input[2].append(i[2])
            self.train_input[3].append(i[3])
        for i in range(4):
            self.train_input[i] = np.array(self.train_input[i])
            self.train_input[i] = np.reshape(self.train_input[i], (len(self.train_input[i]), 1, 10))


        self.train_output = np.array(self.train_label[:l0])

        self.test_input = [[],[],[],[]] 
        for i in self.four_tuples[l0:]:
            self.test_input[0].append(i[0])
            self.test_input[1].append(i[1])            
            self.test_input[2].append(i[2])
            self.test_input[3].append(i[3])
        for i in range(4):
            self.test_input[i] = np.array(self.test_input[i])
            self.test_input[i] = np.reshape(self.test_input[i], (len(self.test_input[i]), 1, 10))


        self.test_output = np.array(self.train_label[l0:])



        self.test_input = self.train_input
        self.test_output = self.train_output


x = DataSet('exp')
print(len(x.four_tuples))