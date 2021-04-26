# Deep Learning Project

### 1. Overview 



### 2. The Dataset

The team originally intended to use the IM2LATEX-100k dataset provided by the original paper, containing 103,556 formulas and their corresponding images of fixed A4 sizes. 

However,  there was a lot of white space in the images as the formulae only took up 1-2 lines. This was a big source of inefficiency in training our model, especially since we need to perform computationally heavy operations for the task. 

Through more research, we found a cleaned dataset by [source here]. The images were processed such that the image size was the size of the formulae (non-uniform size across the dataset). Moreover, their corresponding latex formulae were also normalised by having a "\space" character between each token.

#### 2. 1 Data engineering 

An inversion of colours of the images such that their background colours were black. 

Due to the non-uniform size of images (and corresponding formulae) across the dataset, we encountered difficulties in batching training samples. This was overcome by writing a custom ```collate_fn``` for the torch data loader which:

1. Padded images to the largest image size in its batch. 
2. Padded formulae to the longest length in its batch.

We provide an example: 



|  Processed Image   | <img src="C:\Users\Chan Luo Qi\Documents\SUTD\Term 11\Deep Learning\Big Project\Figure_1.png" style="zoom:0%;" /> |
| :----------------: | :----------------------------------------------------------- |
| **Padded Formula** | \int _ { - \epsilon } ^ { \infty } d l \: \mathrm { e } ^ { - l \zeta } \int _ { - \epsilon } ^ { \infty } d l ^ { \prime } \mathrm { e } ^ { - l ^ { \prime } \zeta } l l ^ { \prime } { \frac { l ^ { \prime } - l } { l + l ^ { \prime } } } \{ 3 \, \delta ^ { \prime \prime } ( l ) - { \frac { 3 } { 4 } } t \, \delta ( l ) \} = 0 . \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos \eos |

#### 2. 2 Dataset Pruning

As the dataset is large and model training is computationally heavy, we only use a subset of the dataset with 30 or less tokens in the formulae. The train-test split is as shown below:

| Data  | #     |
| ----- | ----- |
| Train | 12265 |
| Test  | 616   |

### 3. Model 

#### 3. 1. Encoder: CNN and bi-directional LSTM 

The processed image is passed through some convolutional layers to obtain a feature map. It is then passed through a simple row encoder. The row encoder helps to encode information in the image in left-to-right sequence, similar to how an equation is read in reality. 

[image here]

#### 3. 2. Decoder: Gated Recurrent Unit (GRU) with attention

Predictive alignment is used in our decoder. 

