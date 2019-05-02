import tflearn
from tflearn import input_data, conv_2d, max_pool_2d, fully_connected, regression, dropout, batch_normalization


class BeeClassifierNet:

    def MichaelGeigl(self, tensorWidth, tensorHeight, tensorDepth):

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        conv_net = conv_2d(conv_net,50,4,activation='tanh')
        conv_net = max_pool_2d(conv_net,5)
        conv_net = conv_2d(conv_net,50,4,activation='tanh')
        conv_net = max_pool_2d(conv_net,5)
        conv_net = conv_2d(conv_net,100,4,activation='tanh')
        conv_net = max_pool_2d(conv_net,5)
        conv_net = fully_connected(conv_net, 50, activation='tanh', name='fl1')

        conv_net = fully_connected(conv_net, 2, activation='softmax',name='output')

        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    def DavidSpencer(self, tensorWidth, tensorHeight, tensorDepth):

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        conv_net = conv_2d(conv_net,nb_filter=32,filter_size=5, activation='relu', bias=True)
        conv_net = batch_normalization(conv_net)
        conv_net = max_pool_2d(conv_net, 4)
        conv_net = dropout(conv_net, 0.5)
        conv_net = fully_connected(conv_net, 100, activation='relu')
        conv_net = dropout(conv_net, 0.5)


        conv_net = fully_connected(conv_net, 2, activation='softmax',name='output')
        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model


    def ArihantJain(self, tensorWidth, tensorHeight, tensorDepth):

        network = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network,
                             optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        model = tflearn.DNN(network, tensorboard_verbose=0)

        return model

    def ManishMeshram(self, tensorWidth, tensorHeight, tensorDepth):

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        conv_net = conv_2d(conv_net,
                          nb_filter=20,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_1')
        conv_net = conv_2d(conv_net,
                               nb_filter=25,
                               filter_size=5,
                               activation='relu',
                               name='conv_layer_2')

        conv_net = max_pool_2d(conv_net, 2, name='pool_layer_1')
        conv_net = fully_connected(conv_net, 100,
                                     activation='sigmoid',
                                     name='fc_layer_1')
        conv_net = fully_connected(conv_net, 2,
                                     activation='softmax',
                                     name='fc_layer_2')


        conv_net = fully_connected(conv_net, 2, activation='softmax',name='output')

        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    def AdamKing(self, tensorWidth, tensorHeight, tensorDepth):

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net, nb_filter=10, filter_size=10,
                               activation='tanh', name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 4, name='pool_layer_1')
        conv_net = conv_2d(conv_net, nb_filter=30, filter_size=3,
                               activation='tanh', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2, name='pool_layer_1')
        conv_net = fully_connected(
            conv_net, 100, activation='sigmoid', name='fc_layer_1')



        conv_net = fully_connected(conv_net, 2, activation='softmax',name='output')

        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    def AsherGunsay(self, tensorWidth, tensorHeight, tensorDepth):

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net, nb_filter=20, filter_size=5, activation='relu', name='conv_1')
        conv_net = max_pool_2d(conv_net, 2, name='pool_1')
        conv_net = conv_2d(conv_net, nb_filter=15, filter_size=5, activation='relu', name='conv_2')
        conv_net = max_pool_2d(conv_net, 2, name='pool_2')
        conv_net = fully_connected(conv_net, 15, activation='relu', name='fc_layer_1')

        conv_net = fully_connected(conv_net, 2, activation='sigmoid',name='output')
        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    def RyanWilliams(self, tensorWidth, tensorHeight, tensorDepth):

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net, nb_filter=20,
                             filter_size=7, activation='relu',
                             name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 3, name='pool_layer_1')
        conv_net = conv_2d(conv_net, nb_filter=20,
                             filter_size=5, activation='relu',
                             name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2, name='pool_layer_2')
        conv_net = fully_connected(conv_net, 100,
                                     activation='sigmoid', name='fc_layer_1')


        conv_net = fully_connected(conv_net, 2, activation='sigmoid',name='output')
        conv_net = regression(conv_net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    #Don't have it yet
    def CameronFrandsen(self, tensorWidth, tensorHeight, tensorDepth):

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])


        conv_net = fully_connected(conv_net, 2, activation='softmax',name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    def ChrisKinsey(self, tensorWidth, tensorHeight, tensorDepth):

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net, nb_filter=20, filter_size=5, activation="relu", name="conv_layer_1")
        conv_net = max_pool_2d(conv_net, 2, name="pool_layer")
        conv_net = conv_2d(conv_net, nb_filter=50, filter_size=5, activation="relu", name="conv_layer_2")
        conv_net = max_pool_2d(conv_net, 2, name="pool_layer")
        conv_net = conv_2d(conv_net, nb_filter=30, filter_size=5, activation="relu", name="conv_layer_3")
        conv_net = max_pool_2d(conv_net, 2, name="pool_layer")
        conv_net = fully_connected(conv_net, 140, activation="relu", name="dense_layer_1")
        conv_net = fully_connected(conv_net, 80, activation="sigmoid", name="dense_layer_2")
        conv_net = fully_connected(conv_net, 2, activation="sigmoid", name="dense_layer_3")

        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    def VishalSharma(self, tensorWidth, tensorHeight, tensorDepth):


        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])
        conv_net = conv_2d(conv_net,
                               nb_filter=32,
                               filter_size=3,
                               activation='relu',
                               name='conv_layer_1')
        conv_net = conv_2d(conv_net,
                               nb_filter=64,
                               filter_size=3,
                               activation='relu',
                               name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2, name='pool_layer_1')
        conv_net = fully_connected(conv_net, 256, activation='sigmoid', name='fc_layer_1')
        conv_net = fully_connected(conv_net, 2, activation='sigmoid', name='fc_layer_2')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model
