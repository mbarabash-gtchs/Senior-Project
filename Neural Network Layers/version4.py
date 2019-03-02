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