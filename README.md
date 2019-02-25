# Portfolio of my work/what I learned connected to Neural Networks/Machine Learning while doing my Senior Project
This main document is split into two sections; the first describes the problem/goal of my project, the second describes my approach to it.


## Problem
### Introduction to Neural Networks
To begin, a Neural Network is computer code that is designed to function like the human brain. The network consists of several "layers," which takes a specific type of input data and transforms it into the desired output. Within these layers are "neurons," which are essentially miniature functions that perform non-linearities on the weighted sum of their inputs. In order to reach the correct output, the neurons need their weights to be properly tuned. The value of those weights is calculated through a process called training. 
#### The training process goes as follows:
 * An input is given and is evaluated by the network.
 * The output of the network is evaluated; The “amount” of error made by the network is quantified as a quantity called loss (The ideal weights would produce a minimum loss value).
 * The weights are adjusted slightly to reduce the loss of the network. There are many ways to adjust this loss via different optimizers, but I chose to use the Adam Optimizer. 
As mentioned above, a dataset is needed in order to train the network. However, there are two main conditions that the dataset needs to meet in order to train a working network. These are:
 * Size: Neural Networks are designed to work like the human brain does; this means that the network will take the easiest method to get correct answers. If a dataset that a network is training on is too small, then the network will end up simply memorizing the images and their corresponding answers instead of learning why the answers correspond with images.
 * Variance: If images within the dataset are similar to one another, then the network will struggle to identify images that are different from those it trained on.


### Goal
I worked with Visual Neural Networks specifically, so I will describe how my neural network works as though it were a brain processing what the eyes send it.

Ideally, the process for a visual neural network would go like this:
* 1) Image is fed to the network
* 2) The first few layers of the network will identify the, for lack of a better term, "material" features such as edges, colors, etc. The image below shows an approximation of what an individual neuron may detect.
![alt text](http://cs231n.github.io/assets/cnnvis/filt1.jpeg) 
(Image taken from Andrea Vedaldi and Aravindh Mahendran)

* 3) The layers following the first few will identify how previous layers are interacting with each other, and will therefore be able to identify more abstract features, such as a face, an animal, or whatever.
* 4) The final layers will use the outputs from the middle layers to provide an output that describes whatever the creator of the network wants it to describe.

To elaborate on step 4: If I wanted the network to tell me whether a picture is of a cat or of a dog, then it would do that. If I wanted the network to do something more fancy, such as edit the output image, I could make it do that as well.



### Final Product
My final product is a neural network that detects the position of a dogs tail. This means that the network would need to find a dog in an image, find the tail of the dog, and then find how the tail is positioned relative to the dog. Theoretically, if a network can detect the position of a tail relative to a dog, then it would be able to detect other objects relative to another. The applications of such a network can be seen clearly; for example, a network that has been trained on MRI scans could learn to detect brain tumors. 

### Prerequisites for Final Product

## Approach to Problem

#### Building a dataset
Building a dataset was actually something that I ended up spending a lot of time on. As I described above, training a neural network requires multiple inputs; however, an obvious problem is that if there are not many 
#### Learn TensorFlow
Learning TensorFlow (an open-source software library designed for handling data) would be something that I would do throughout the coding process. There is no easy way to directly learn TensorFlow that I am aware of; the only way is through application.
