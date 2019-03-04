Note: all coding was done within jupyter. I have uploaded the source code for my layers of the neural network here, but their implementation can be seen in the Takeaway directory.

Also please note: "model" refers to the appended layers themselves. "Full model" refers to both the appended layers as well as the code that trains the model.

## Variants of Network:

#### Version 1
Version 1 had a problem; the full model would begin training, but would then the loss, as shown below, would jump around and stagnate. 

![alt text](https://i.imgur.com/MccPCPm.png)

This version was atrocious. Not only did the full model not learn, but it was not even able to overfit a small dataset, meaning that something fundamental was wrong. I built several variants of the full model, and each seemed to provide small improvements. I started shuffling the order that pictures were presented to the model, resulting in a nicer looking loss decrease. The loss would still eventually stop decreasing at an alarmingly large number, so I decided to see if there was something wrong with my dataset instead. It turned out that my dataset, while creating variance, did not correctly change the labels of each corresponding picture. After I resolved that problem, version 1 was able to overfit a dataset of 9 pictures, shown below. This was still far from a satisfactory result however; learning 9 images should be nothing. 

![alt text](https://i.imgur.com/q8YHV5c.png)

#### Version 2
Version 2 seemed to be an improvement to Version 1 at first; it was able to overfit small datasets much quicker than Version 1. However, Version 2 was still unable to overfit larger datasets.

#### Version 3
Version 3's full model results were practically not much better, if not worse, than the version 2 full model. It is difficult to judge because both version 2 and version 3 were unable to reach a satisfactory fit accuracy. 
The changes performed were to the appended model. In version 3, the size of each layer was slightly different, although the total amount of layers remained the same. The following graph shows the trend that both version 2 and 3 tended to follow while training larger datasets.

![alt text](https://i.imgur.com/mEHsQai.png)

#### Version 4
Version 4 made no immediate changes to appended layers; the only changes were to the full model. Version 4 utilized version 2's appended layers, and simply changed the way that training occurred. The adjustment that caused version 4 to perform well was the introduction of normalization. Every input would be normalized prior to being evaluated by the network, which allows the network to work better. Normalization was done by taking the average of all inputs and then dividing by the standard deviation.

![alt text](https://i.imgur.com/w2iNRaF.png)

I eventually changed the appended layers as well. One of the networks was deeper and had more layers (named Version 4 extended), while the other used more average pools instead of convolutions (named Version 4 average pool). I did not get significant changes in results from either of them.

#### Version 5
Version 5 used a different approach than prior versions; instead of taking the size of the layers and decreasing them slowly, it instead changed the size of the layers to be 13x13x18 as quickly as possible and then remained at that size for 6 layers prior to being converted to the output size. This, for some reason, caused the validation accuracy to rise to higher values than I had previously recorded.