
def appended_model_v1(XB, reuse=True):
    '''input: XB - input from layer B of squeezenet
    returns: layers[0...6]'''
    x=XB
    layers=[]
    var_list=[] # list of variable tensors so that we can initialize variables for a particular model
    with tf.variable_scope('my_v1', reuse=reuse):
        with tf.variable_scope('layer0'):
            W = tf.get_variable("weights",shape=[1,1,64,18])
            b = tf.get_variable("bias",shape=[18])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 1:
            x = tf.nn.relu(x)
            layers.append(x)
        #
        # fully connected branch:
        #
        with tf.variable_scope('layer2'):
            W = tf.get_variable("weights",shape=[1,1,18,18])
            b = tf.get_variable("bias",shape=[18])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 3:
            x = tf.nn.relu(x)
            layers.append(x)
        with tf.variable_scope('layer4'):
            W = tf.get_variable("weights",shape=[1,1,18,6])
            b = tf.get_variable("bias",shape=[6])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 5:
            x = tf.nn.relu(x)
            layers.append(x)
        #
        #fully connected layer:
        with tf.variable_scope('layer6'):
            W = tf.get_variable("weights",shape=[13,13,6,6])
            b = tf.get_variable("bias",shape=[6])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,13,13,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
    return layers, var_list
                
                