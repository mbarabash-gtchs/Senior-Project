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

To elaborate on step 4: If I wanted the network to tell me whether a picture is of a cat or of a dog, then it would do that. If I wanted the network to do something more fancy, such as edit the output image, I could make it do that as well.

How is this final stage reached? Well, after the skeleton of the neural network is built, the initial weights of the machine mean nothing. The network is then given an input, and it produces an output which is either correct or incorrect to a specific degree. Based off of this, the weights of the individual neurons are adjusted and the process is repeated with a different input.

The goal of my project was to train my neural net to recognize the location of a dogs tail. The reasoning behind this was that neural networks that classify objects are numerous, and I worked with them a bit while studying the basics of neural networks. I chose to do something a little different to up the difficulty of the project.

## Prerequisites for Final Product
In terms of my prior knowledge: I had known a decent amount about neural networks from studying them over the summer. However, I only knew about the theory behind it; to give an analogy, if I were building a house, then I knew how to create a blueprint; I needed to learn how to physically build the house.
Before I could begin working on my final product, there were several things I needed to do. 

#### Building a dataset
Building a dataset was actually something that I ended up spending a lot of time on. As I described above, training a neural network requires multiple inputs; however, an obvious problem is that if there are not many 
#### Learn TenserFlow
Learning TenserFlow (an open-source software library designed for handling data) would be something that I would do throughout the coding process. There is no easy way to directly learn TenserFlow that I am aware of; the only way is through application.
