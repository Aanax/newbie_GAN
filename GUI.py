import sys
from PyQt5.QtWidgets import QWidget, QCheckBox, QHBoxLayout, QDesktopWidget, QApplication, QMessageBox, QPushButton, QLabel, QProgressBar, QTextEdit, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QCoreApplication,  pyqtSignal
from PyQt5.QtCore import QThread, Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication
#import design
import os
import numpy as np
from sklearn.externals import joblib
from PIL import Image
import webbrowser

class netLearningThread(QThread):
    '''
    Class used to train the net in separate thread
    '''

    transitional_pic = pyqtSignal(QPixmap)
    idol_pic = pyqtSignal(QPixmap)
    progress = pyqtSignal(int)
    saver = pyqtSignal(str)

    def __init__(self, usesaved, savename, n_epoch=2):
        QThread.__init__(self)

        self.savename = savename
        self.n_epoch = n_epoch
        self.usesaved = usesaved

    def __del__(self):
        self.wait()

    def showpic(self, digit, multiplier=70, size=200, shape=(8, 8)):
        '''
        Converts np.array into Qpixmap. Whith desired resizing and multiplying
        :param digit: np.array(image), must be broadcastable to shape shape
        :param multiplier: param to be multiplied with digit (brightness correction)
        :param size: digit will be resized to (size,size)
        :param shape: digit must be broadcastable to shape shape
        :return: Qpixmap
        '''
        tstimage1 = Image.fromarray(digit.reshape(shape) * multiplier)
        tstimage1 = tstimage1.convert('L')
        tstimage1 = tstimage1.resize((size, size))
        qPix = tstimage1.toqpixmap()
        #data = tstimage1.toString('raw', 'RGBA')
        #qIm = QtGui.QImage(data, tstimage1.size[0], tstimage1.size[1], QtGui.QImage.Format_ARGB32)
        return qPix
        # tstimage1.show()

    def train_net(self, usesaved, n_epoch=500):
        '''
        The whole training done here. Defines the net. Loads data. Trains the net.
        Rewrites net weights into file each epoch.
        :param usesaved: Whether to use precomputed weights from file
        :param n_epoch: Number of epochs to learn
        :return:
        '''

        from keras.models import Model, Sequential
        from keras.layers import Flatten, Dropout, Dense, Input, merge
        from sklearn.datasets import load_digits


        # DATA

        X, y = load_digits(n_class=1, return_X_y=True)

        # showpic(X[0])

        # MODEL
        gen_input = Input(shape=(64,))
        orig_input = Input(shape=(64,))

        generator = Dense(64, activation='sigmoid')
        # generator2 = Dense(64,activation = 'sigmoid')
        # gen_out1 = generator(gen_input)
        gen_out = generator(gen_input)

        merged_vector = merge([orig_input, gen_out], mode='sum', output_shape=(64,))

        predictions = Dense(1, activation='sigmoid')(merged_vector)

        model = Model(input=[gen_input, orig_input], output=predictions)

        from theano.tensor import basic as tensor, subtensor
        def generator_error_theano(y_true, y_pred):
            '''
            TODO write normalisation
            '''
            return -tensor.log(1.0 - y_pred)  # -(target * tensor.log(output) + (1.0 - target) *
            # return -tensor.log(1.0 + (y_pred-1.0))

        def tester_error_theano(target, output):
            '''
            y_true = 0 -> it was real
            '''

            return -((target) * tensor.log(output) + (1.0 - target) * tensor.log(1.0 - output))

        npx = np.array(X)

        bottom = np.array(X).min()
        top = np.array(X).max()

        X = X / top

        top = np.array(X).max()

        idol = self.showpic(X[np.random.choice(len(X))])
        self.idol_pic.emit(idol)


        generator_layers = 1
        tester_train_steps = 1
        # n_epoch = 500
        xnum = 0

        #self.saver.connect(joblib.dump(model.get_weights(),"qwe.pkl"))

        from keras.optimizers import SGD
        opt = SGD(lr=0.0001)

        import theano
        get_activations = theano.function([model.layers[0].input], model.layers[1].output, allow_input_downcast=True)

        # LOAD WEIGHTS
        if self.usesaved:
            model.set_weights(joblib.load(self.savename))

        # TRAINING LOOP

        for epoch in range(0, n_epoch):

            print("Epoch ", epoch)  # ,"Iter ",xnum)
            joblib.dump(model.get_weights(), self.savename, compress=9)
            for xnum in range(0, len(X)):

                # unfreezing tester
                model.layers[-1].trainable = True
                # freezing generator
                for i in range(1, 1 + generator_layers + 1):
                    model.layers[i].trainable = False
                model.compile(opt, loss=tester_error_theano)
                # doing tester_train_steps of tester training
                for k in range(0, tester_train_steps):
                    # for xnum in range(0,len(X)):
                    model.train_on_batch([np.zeros((64,)).reshape((-1, 64)), np.array(X[xnum]).reshape((-1, 64))],
                                         np.array([0.0]))  # inputs original image
                    model.train_on_batch(
                        [np.random.uniform(bottom, top, 64).reshape((-1, 64)), np.zeros((64,)).reshape((-1, 64))],
                        np.array([1.0]))  # inputs noise to generator

                # Unfreezing generator
                for i in range(1, 1 + generator_layers + 1):
                    model.layers[i].trainable = True
                # freezing tester
                model.layers[-1].trainable = False
                model.compile(opt, loss=generator_error_theano)
                # doing ? steps of generator training
                for nn in range(0, 3):
                    model.train_on_batch(
                        [np.random.uniform(bottom, top, 64).reshape((-1, 64)), np.zeros((64,)).reshape((-1, 64))],
                        np.array([1.0]))  # inputs noise to generator
                if xnum % 10 == 0: #len(X) - 1:
                    tst = np.random.uniform(bottom, top, 64).reshape((-1, 64))
                    Generated = get_activations(tst)
                    pixmap = self.showpic(Generated) # Image.fromarray(Generated.astype(np.uint8))
                    # pixmap = QPixmap.fromImage(im)
                    self.transitional_pic.emit(pixmap)
                    # showpic(Generated, size=500, multiplier=70)

            self.progress.emit(epoch)

        return 1# "'{title}' by {author} in {subreddit}".format(**top_post)

    def run(self):
        '''
        Default method. Everything in thread runs from here.
        :return:
        '''
        self.train_net(self.usesaved, self.n_epoch)



class Example(QWidget,QThread):
    '''
    Main window class. Contains all widgets and communicates with
    net training thread
    '''

    def __init__(self):
        super().__init__()

        self.initUI()

    def runtrain(self):
        '''
        Creates net training thread and runs.
        :return:
        '''
        n_epoch = 200
        usesaved = self.savebox.isChecked()
        savepath = self.saveedit.toPlainText()
        self.get_thread = netLearningThread(usesaved, savepath, n_epoch)
        self.get_thread.transitional_pic.connect(
            self.writeSmth)  # self.get_thread, SIGNAL('transitional_pic(Qmatrix?)'),
        self.get_thread.idol_pic.connect(
            self.show_idol)
        self.pbar.setMaximum(n_epoch-1)
        self.get_thread.progress.connect(self.pbar.setValue)

        self.get_thread.start()

    def show_idol(self, Smth):
        '''
        Draws the picture of desired result.
        :param Smth: Desired result (Qpixmap)
        :return:
        '''
        # self.label.setText("GotSmth")
        # pixmap = QPixmap(os.getcwd() + '/cutted.jpg')
        myScaledPixmap = Smth.scaled(self.label_idol.size(), Qt.KeepAspectRatio)
        self.label_idol.setPixmap(myScaledPixmap)

    # def saveme(self):
    #
    #     self.aa = QVBoxLayout()
    #     self.savefield = QTextEdit()
    #     self.okbtn = QPushButton('OK', self)
    #     #self.savefield.show()
    #     self.aa.addWidget(self.okbtn)
    #     self.aa.addWidget(self.savefield)
    #     self.label111 = QLabel()
    #     self.label111.setLayout(self.aa)
    #     self.label111.show()
    #     self.okbtn.clicked.connect(self.emitsave)

    # def emitsave(self):
    #     path = self.savefield.toPlainText()
    #     self.get_thread.saver.emit(str(path))



        #QTextEdit.keyPressEvent()

    def helpopen(self):
        '''
        opens helpfile in browser
        :return:
        '''
        webbrowser.open("./help.htm")

    def initUI(self):
        '''
        Creates widgets.
        :return:
        '''

        self.setGeometry(1300, 300, 400, 400)
        self.setWindowTitle('Message box')

        OneVert = QVBoxLayout()
        TwoVert = QVBoxLayout()
        SaveHor = QVBoxLayout()
        BigHor = QHBoxLayout(self)

        qbtn = QPushButton('start', self)
        qbtn.clicked.connect(self.runtrain)
        qbtn.setMaximumWidth(200)
        #qbtn.clicked.connect(self.runtrain)
        # qbtn.resize(qbtn.sizeHint())
        #qbtn.move(50, 50)

        # savebtn = QPushButton('save', self)
        # savebtn.clicked.connect(self.saveme)
        # savebtn.move(200, 50)

        helpbtn = QPushButton("Help",self)
        helpbtn.clicked.connect(self.helpopen)
        helpbtn.setMaximumWidth(200)


        self.savebox = QCheckBox(self)
        self.savebox.setText("UseSaved")

        self.label = QLabel("LABEL",self)
        #self.label.move(100,100)
        self.label.resize(200, 200)

        self.label_idol = QLabel("LABEL_IDOL", self)
        #self.label_idol.move(500, 100)
        self.label_idol.resize(200, 200)

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(200, 80, 250, 20)
        self.pbar.setMaximumWidth(200)
        #self.pbar.move(300,420)

        # pixmap = QPixmap(os.getcwd() + '/cutted.jpg')
        # self.label.setPixmap(pixmap)

        self.saveedit = QTextEdit("Save/load path", self)
        self.saveedit.setMaximumWidth(200)
        self.saveedit.setMaximumHeight(60)

        #OneVert.setGeometry( 0,0,500,500)
        OneVert.addWidget(qbtn)
        #OneVert.addWidget(self.savebox)
        OneVert.setAlignment(qbtn, Qt.AlignTop)
        SaveHor.addWidget(self.savebox)
        SaveHor.addWidget(self.saveedit)
        OneVert.addLayout(SaveHor)
        OneVert.setAlignment(SaveHor, Qt.AlignTop)
        OneVert.addWidget(self.pbar)
        OneVert.setAlignment(self.pbar, Qt.AlignTop)

        OneVert.addWidget(helpbtn)
        #OneVert.setAlignment(OneVert.widget(), Qt.AlignTop)

        TwoVert.addWidget(self.label)
        TwoVert.addWidget(self.label_idol)

        BigHor.addLayout(OneVert)
        BigHor.addLayout(TwoVert)
        self.setLayout(BigHor)

        self.show()


    def writeSmth(self, Smth):
        '''
        Updates the pic of current progress.
        :param Smth: pic of current progress (Qpixmap)
        :return:
        '''
        # self.label.setText("GotSmth")
        # pixmap = QPixmap(os.getcwd() + '/cutted.jpg')
        myScaledPixmap = Smth.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(myScaledPixmap)

    def closeEvent(self, event):
        pass
        # reply = QMessageBox.question(self, 'Message',
        #   "Are you sure to quit?", QMessageBox.Yes |
        #   QMessageBox.No, QMessageBox.No)

        # if reply == QMessageBox.Yes:
        #   event.accept()
        # else:
        #   event.accept()

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())