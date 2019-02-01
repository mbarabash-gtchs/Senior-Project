# Portfolio of my work/what I learned connected to Neural Networks/Machine Learning while doing my Senior Project

## Introduction to Neural Networks
To begin, a Neural Networks is computer code that is designed to function like the human brain. The network consists of several "layers" which a specific type of input data and output a specific type of output. Within these layers are "neurons," which are essentially miniature functions the perform weighted non-linearities on the input.


### Example of non-linearity
![alt text](https://cdn-images-1.medium.com/max/1600/1*DfMRHwxY1gyyDmrIAd-gjQ.png)


## Goal
I worked with Visual Neural Networks specifically, so I will describe how my neural network works as though it were a brain processing what the eyes send it. 

Ideally, the process for a visual neural network would go like this:
* 1) Image is fed to the network
* 2) The first few layers of the network will identify the, for lack of a better term, "material" features such as edges, colors, etc.
* 3) The layers following the first few will identify how these layers are interacting with each other, and will identify more abstract features, such as a face, an animal, or whatever. 
* 4) The final layers will use the outputs from the middle layers to provide an output that describes whatever the creator of the network wants it to describe. 

To elaborate on 4: If I wanted the network to tell me whether a picture is of a cat or of a dog, then it would do that. If I wanted the network to do something more fancy, such as edit the output image, I could make it do that as well.

The goal of my project was to train my neural net to recognize the location of a dogs tail. The reasoning behind this was that neural networks that classify objects are numerous, and I worked with them a bit while studying neural networks. I chose to do something a little different to up the difficulty of the project.
