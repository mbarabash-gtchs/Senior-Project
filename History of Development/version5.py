def appended_model_v5(XB,XC,XD, reuse=True):
    # same as appended_model_v4_avgpool
    '''input: XB,XC, XD - input from layers B,C,D of squeezenet
    returns: layers[0...6]'''
    layers=[]
    var_list=[] # list of variable tensors so that we can initialize variables for a particular model
    with tf.variable_scope('my_v2', reuse=reuse):
        x=XD #55x55x16
        with tf.variable_scope('layer0-D'):
            W = tf.get_variable("weights",shape=[1,1,16,10])
            b = tf.get_variable("bias",shape=[10])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 1:
            x = tf.nn.relu(x)
            layers.append(x)
            #layer 2:
            x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
            layers.append(x)
            # x now is 27x27x10
            
        #this now has the same HW dimension as layer C (which is 27x27x32) :
        x = tf.concat([x,XC], 3) #3 is axis   
        with tf.variable_scope('layer3-C'):
            W = tf.get_variable("weights",shape=[1,1,42,12])
            b = tf.get_variable("bias",shape=[12])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 4:
            x = tf.nn.relu(x)
            layers.append(x)
            #layer 5:
            x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
            layers.append(x)
            # x now is 13x13x12
        #XB is  13x13x64
        x  = tf.concat([x,XB], 3)   
        with tf.variable_scope('layer6-B'):
            W = tf.get_variable("weights",shape=[1,1,76,18])
            b = tf.get_variable("bias",shape=[18])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 7:
            x = tf.nn.relu(x)
            layers.append(x)
        #x is 13x13x18
        with tf.variable_scope('layer8'):
            W = tf.get_variable("weights",shape=[1,1,18,18])
            b = tf.get_variable("bias",shape=[18])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 9:
            x = tf.nn.relu(x)
            layers.append(x)
        #x has shape 13x13x18
        
        with tf.variable_scope('layer10'):
            W = tf.get_variable("weights",shape=[1,1,18,18])
            b = tf.get_variable("bias",shape=[18])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 11:
            x = tf.nn.relu(x)
            layers.append(x)
        #
        #averagepool:
        with tf.variable_scope('layer12'):
            x = tf.nn.avg_pool(x,[1,13,13,1],strides=[1,1,1,1],padding = "VALID")
            layers.append(x)
            W = tf.get_variable("weights",shape=[1,1,18,6])
            b = tf.get_variable("bias",shape=[6])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            layers.append(x)
    return layers, var_list
                
  