def tftonp(filename, batch_size):
    
    
    ds=dataset_XY_from_TFRecord( layer_name_and_shape_list=[    ("squeezed_layer_1", 55, 55, 16), 
                                                                ("squeezed_layer_3", 27, 27, 32),
                                                                ("squeezed_layer_7", 13, 13, 64)  ], filename=filename )
    
    dataset = ds.repeat().batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    dctn = sess.run(next_element)
    XB,XC,XD = dctn['squeezed_layer_1'],dctn['squeezed_layer_3'],dctn['squeezed_layer_7']
    
    
    averageb,stdb = normalize_num(XB)
    averagec,stdc = normalize_num(XC)
    averaged,stdd = normalize_num(XD)
    return averageb,stdb,averagec,stdc,averaged,stdd

    
def normalize_num(layer):
    average = np.mean(layer, axis=(0,1,2))
    std = np.std(layer, axis=(0,1,2))
    return average, std