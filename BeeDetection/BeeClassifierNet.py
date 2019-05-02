import tflearn
from tflearn import input_data, conv_2d, max_pool_2d, fully_connected, regression, dropout, batch_normalization


class BeeClassifierNet:

    def Model1(self, tensorWidth, tensorHeight, tensorDepth):
        # aug = tflearn.ImageAugmentation()
        # aug.add_random_blur(0.8)
        # aug.add_random_rotation(12)

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth])

        conv_net = conv_2d(conv_net,
                           nb_filter=16,
                           filter_size=3,
                           activation='relu', name='conv_layer_2',regularizer='L2')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)

        conv_net = conv_2d(conv_net,
                           nb_filter=32,
                           filter_size=3,
                           activation='relu', name='conv_layer_2',regularizer='L2')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=64,
                           filter_size=3,
                           activation='relu', name='conv_layer_3',regularizer='L2')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=128,
                           filter_size=3,
                           activation='relu', name='conv_layer_3',regularizer='L2')
        conv_net = max_pool_2d(conv_net, 2)


        conv_net = dropout(conv_net,keep_prob=0.5)
        conv_net = fully_connected(conv_net, 1024, activation='relu',name='fc_layer_1')

        conv_net = dropout(conv_net,keep_prob=0.5)
        conv_net = fully_connected(conv_net, 1024, activation='relu',name='fc_layer_2')


        conv_net = fully_connected(conv_net, 2, activation='softmax',name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    def Model2(self, tensorWidth, tensorHeight, tensorDepth):
        aug = tflearn.ImageAugmentation()
        aug.add_random_blur(3)

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth],data_augmentation=aug)

        conv_net = conv_2d(conv_net,
                           nb_filter=64,
                           filter_size=3,
                           activation='relu', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=128,
                           filter_size=3,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=256,
                           filter_size=3,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)


        conv_net = dropout(conv_net,keep_prob=0.5)
        conv_net = fully_connected(conv_net, 1024, activation='relu',name='fc_layer_1')

        conv_net = dropout(conv_net,keep_prob=0.5)
        conv_net = fully_connected(conv_net, 1024, activation='relu',name='fc_layer_2')


        conv_net = fully_connected(conv_net, 2, activation='softmax',name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    #BEst accuracy so far with 4 fc - 93.48%
    def Model4(self, tensorWidth, tensorHeight, tensorDepth):
        aug = tflearn.ImageAugmentation()
        # aug.add_random_blur(3)

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth], data_augmentation=aug)

        conv_net = conv_2d(conv_net,
                           nb_filter=64,
                           filter_size=3,
                           activation='relu', name='conv_layer_1')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=128,
                           filter_size=3,
                           activation='relu', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=256,
                           filter_size=3,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)

        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 512, activation='relu', name='fc_layer_1')

        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 256, activation='relu', name='fc_layer_2')
        #
        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 128, activation='relu', name='fc_layer_3')

        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 64, activation='relu', name='fc_layer_4')

        conv_net = fully_connected(conv_net, 2, activation='softmax', name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model

    #trained for 25 epochs with a validation acc of 92.6
    def Model3(self, tensorWidth, tensorHeight, tensorDepth):
        aug = tflearn.ImageAugmentation()
        aug.add_random_blur(3)

        conv_net = input_data(shape=[None, tensorWidth, tensorHeight, tensorDepth], data_augmentation=aug)

        conv_net = conv_2d(conv_net,
                           nb_filter=64,
                           filter_size=3,
                           activation='relu', name='conv_layer_2')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=128,
                           filter_size=3,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)
        conv_net = batch_normalization(conv_net)
        conv_net = conv_2d(conv_net,
                           nb_filter=256,
                           filter_size=3,
                           activation='relu', name='conv_layer_3')
        conv_net = max_pool_2d(conv_net, 2)

        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 512, activation='relu', name='fc_layer_1')

        conv_net = dropout(conv_net, keep_prob=0.5)
        conv_net = fully_connected(conv_net, 256, activation='relu', name='fc_layer_2')

        conv_net = fully_connected(conv_net, 2, activation='softmax', name='output')
        conv_net = regression(conv_net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy',
                              name='targets')

        model = tflearn.DNN(conv_net, tensorboard_verbose=0)

        return model