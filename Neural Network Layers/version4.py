def appended_full_model_v4(filename, averages, batch_size=2, reuse=False):
    
    #no coordinates
    ds=dataset_XYclassif_from_TFRecord( layer_name_and_shape_list=[    ("squeezed_layer_1", 55, 55, 16), 
                                                                ("squeezed_layer_3", 27, 27, 32),
                                                                ("squeezed_layer_7", 13, 13, 64)  ], filename=filename )
    dsbatched=ds.repeat().shuffle(4600).batch(batch_size)
    iterator= dsbatched.make_one_shot_iterator()
    nextdict=iterator.get_next()
    a1,s1,a3,s3,a7,s7=averages
    layers, var_list = appended_model_v2( ((nextdict['squeezed_layer_7']-a7)/s7), 
                                          ((nextdict['squeezed_layer_3']-a3)/s3),
                                          ((nextdict['squeezed_layer_1']-a1)/s1), reuse )
    y=tf.reshape(layers[-1],[-1,6])
    y_vert = y[:,:3]
    y_horiz = y[:,3:]
    
    loss_vert = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(nextdict['class_vert'], 3), 
                                                                        logits=y_vert
                                                                       ))
    loss_horiz = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(nextdict['class_horiz'], 3), 
                                                                        logits=y_horiz
                                                                       ))
    loss = loss_vert + loss_horiz
    
    return (y_vert,y_horiz), (nextdict['class_vert'],nextdict['class_horiz']), loss, var_list
	

def appended_model_v4_extended(XB,XC,XD, reuse=True):
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
            W = tf.get_variable("weights",shape=[1,1,42,24])
            b = tf.get_variable("bias",shape=[24])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 4:
            x = tf.nn.relu(x)
            layers.append(x)
            #layer 5:
            x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,1,1,1],padding='VALID')
            layers.append(x)
            # x now is 25x25x12
            
        with tf.variable_scope('layerc-b'):
            W = tf.get_variable("weights",shape=[1,1,12,12])
            b = tf.get_variable("bias",shape=[12])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)
            #layer 6:
            x = tf.nn.relu(x)
            layers.append(x)
            #layer 7:
            x = tf.nn.max_pool(x,[1,13,13,1],strides=[1,1,1,1],padding='VALID')
            layers.append(x)
        
        
        #XB is  13x13x64
        x  = tf.concat([x,XB], 3)   
        with tf.variable_scope('layer6-B'):
            W = tf.get_variable("weights",shape=[1,1,76,24])
            b = tf.get_variable("bias",shape=[24])
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
            W = tf.get_variable("weights",shape=[1,1,24,18])
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
        #fully connected layer:
        with tf.variable_scope('layer12'):
            W = tf.get_variable("weights",shape=[1,1,18,6])
            b = tf.get_variable("bias",shape=[6])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,13,13,1],"VALID")
            x = tf.nn.bias_add(x,b)
            layers.append(x)

        with tf.variable_scope('layer13'):
            W = tf.get_variable("weights",shape=[1,1,6,6])
            b = tf.get_variable("bias",shape=[6])
            var_list.append(W)
            var_list.append(b)
            x = tf.nn.conv2d(x,W,[1,13,13,1],"VALID")
            x = tf.nn.bias_add(x,b)
            x = tf.nn.relu(x)
            layers.append(x)

            
            
    return layers, var_list
                

def appended_model_v4_avgpool(XB,XC,XD, reuse=True):
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
                
  