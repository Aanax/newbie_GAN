import sys
from PyQt5.QtWidgets import QWidget, QFileDialog, QCheckBox, QHBoxLayout, QDesktopWidget, QApplication, QMessageBox, QPushButton, QLabel, QProgressBar, QTextEdit, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QCoreApplication,  pyqtSignal
from PyQt5.QtCore import QThread, Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication
#import design
import os
import sip
import PyQt5
import numpy as np
from sklearn.externals import joblib
from PIL import Image
import webbrowser
from tqdm import tqdm
import unittest

class netLearningThread(QThread):
    '''
    Class used to train the net in separate thread
    '''

    transitional_pic = pyqtSignal(QPixmap)
    idol_pic = pyqtSignal(QPixmap)
    progress = pyqtSignal(int)
    saver = pyqtSignal(str)

    def __init__(self, savepath, loadpath, n_epoch=2):
        QThread.__init__(self)

        self.savename = savepath
        self.n_epoch = n_epoch
        self.loadpath = loadpath

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

    def train_net(self, n_epoch=500):
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
        if self.loadpath:
            model.set_weights(joblib.load(self.loadpath))

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
        self.train_net(self.n_epoch)



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
        # usesaved = self.savebox.isChecked()
        # savepath = self.saveedit.toPlainText()
        self.get_thread = netLearningThread(self.savep[0],self.loadp[0], n_epoch)
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
        webbrowser.open("/home/aanax/Desktop/GAN/gui/_build/html/index.html")

    def savepath(self):
        '''
        Allows user to choose where to save the model via FileDialog
        '''
        self.savep = QFileDialog.getSaveFileName(filter="*.pkl")
        self.saveedit.setText("Saving to " + str(self.savep[0]) + "\n\n" + "Loading from " + str(self.loadp[0]))
    def loadpath(self):
        '''
        Allows user to choose where to save the model via FileDialog
        '''
        self.loadp = QFileDialog.getOpenFileName(filter="*.pkl")  #OpenFileName(filter="*.pkl")
        self.saveedit.setText("Saving to "+str(self.savep[0])+"\n\n"+"Loading from "+str(self.loadp[0]))
    def initUI(self):
        '''
        Creates widgets.
        '''

        self.setGeometry(1300, 300, 400, 400)
        self.setWindowTitle('Message box')

        self.savep = ["Default.pkl"]
        self.loadp = ["Default.pkl"]

        OneVert = QVBoxLayout()
        TwoVert = QVBoxLayout()
        SaveHor = QVBoxLayout()
        BigHor = QHBoxLayout(self)

        qbtn = QPushButton('Start', self)
        qbtn.clicked.connect(self.runtrain)
        qbtn.setMaximumWidth(200)

        l = QPushButton("Load from ...",self)
        s = QPushButton("Save to ...",self)
        l.setMaximumWidth(200)
        s.setMaximumWidth(200)
        s.clicked.connect(self.savepath)
        l.clicked.connect(self.loadpath)

        #qbtn.clicked.connect(self.runtrain)
        # qbtn.resize(qbtn.sizeHint())
        #qbtn.move(50, 50)

        # savebtn = QPushButton('save', self)
        # savebtn.clicked.connect(self.saveme)
        # savebtn.move(200, 50)

        helpbtn = QPushButton("Help",self)
        helpbtn.clicked.connect(self.helpopen)
        helpbtn.setMaximumWidth(200)


        #self.savebox = QCheckBox(self)
        #self.savebox.setText("UseSaved")

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
        self.saveedit.setMaximumHeight(200)
        self.saveedit.setReadOnly(True)

        #OneVert.setGeometry( 0,0,500,500)
        OneVert.addWidget(qbtn)
        OneVert.addWidget(s)
        OneVert.addWidget(l)
        #OneVert.addWidget(self.savebox)
        OneVert.setAlignment(qbtn, Qt.AlignTop)
        #SaveHor.addWidget(self.savebox)
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

        '''
        # self.label.setText("GotSmth")
        # pixmap = QPixmap(os.getcwd() + '/cutted.jpg')
        myScaledPixmap = Smth.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(myScaledPixmap)

    def closeEvent(self, event):
        sip.delete(self)
        # reply = QMessageBox.question(self, 'Message',
        #   "Are you sure to quit?", QMessageBox.Yes |
        #   QMessageBox.No, QMessageBox.No)

        # if reply == QMessageBox.Yes:
        #   event.accept()
        # else:
        #   event.accept()
if __name__ == '__main__':

    app = QApplication(sys.argv)

    def tst_showpic(digit, multiplier=400.0, size=200, shape=(8, 8)):
        '''
        Copy of showpic function (for test).
        '''
        if (shape[0] == 0) or (shape[1] == 0):
            # print("Zero shape passed")
            return 0
        tstimage1 = Image.fromarray(digit.reshape(shape) * float(multiplier))
        tstimage1 = tstimage1.convert('L')
        tstimage1 = tstimage1.resize((size, size))
        qPix = tstimage1  # .toqpixmap()
        # data = tstimage1.toString('raw', 'RGBA')
        # qIm = QtGui.QImage(data, tstimage1.size[0], tstimage1.size[1], QtGui.QImage.Format_ARGB32)
        return qPix

    def test_showpic(start=0, stop=10, mstart=0, mstop=50):
        '''
        Tests showpic function.
        '''
    # 400
    # 200
        print("Testing showpic. n from ",start," to ",stop," Mult from ",mstart," to ",mstop)
        for n in tqdm(range(start, stop)):
            for mult in np.linspace(mstart, mstop, (mstop - mstart) * 1000):
                shape = (n, n)
                size = n
                X = np.zeros(shape)
                if n != 0:
                    posx = np.random.randint(low=0, high=n)
                    posy = np.random.randint(low=0, high=n)
                    X[posx][posy] = 1.0
                res = tst_showpic(X, multiplier=mult, size=size, shape=shape)
                if n == 0:
                    assert res == 0, ("Passed size 0 Wanted 0 Got", res, "n=", n, "mult=", mult)
                else:
                    assert np.array_equal((X * mult).astype(int).nonzero(), np.array(res).nonzero()), (
                    "Wanted", (X * mult).astype(int).nonzero(), "Got", np.array(res).nonzero(), "n=", n, "mult=", mult)
        #print("All tests gone okay ( n in range", start, "-", stop, "mult in range", mstart, "-", mstop)
        print("Showpic function test - OK")

    class SimpleWidgetTestCase(unittest.TestCase):
        '''
        Class for testing gui.
        '''
        def setUp(self):
            '''
            Creates widgets to test.
            '''
            self.widget = Example()

        def test_default_widget_size(self):
            '''
            Tests main window default size.
            '''
            self.assertEqual(self.widget.size(), PyQt5.QtCore.QSize(400, 400),
                             'incorrect default size')
            print("Default size - OK")

        def test_children(self):
            '''
            Tests whether all widgets was spawned.
            :return:
            '''
            self.assertEqual(len(self.widget.children()), 9 , "Incorrect number of widgents created")
            print("Widgets creation - OK")

        def test_threading_ability(self):
            '''
            Tests whether users computer supports multithreading.
            :return:
            '''
            self.assertGreater(self.widget.thread().idealThreadCount(), 2,
                            "Cant run two threads or the number of processor cores could not be detected.")
            print("Threading ability - OK")

        def tearDown(self):
            '''
            Closes widget that was opened for testing.
            :return:
            '''

            #del self.widget
            #self.widget.hide()
            #self.widget.destroy()
            #self.widget.close()
            #sip.delete(self.widget)
            #self.widget = None
            self.widget.close()
            #self.widget.deleteLater()
            #self.widget.closeEvent()
            #self.widget.destroy() #dispose()




    tester=SimpleWidgetTestCase()
    tester.setUp()
    tester.test_default_widget_size()
    tester.test_children()
    tester.test_threading_ability()
    tester.tearDown()
    #del tester

    #test_showpic()



    ex = Example()
    sys.exit(app.exec_())