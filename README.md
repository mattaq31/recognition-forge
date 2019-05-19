# Forging the Perfect Recognition
Hello and welcome to our project repository. 

We are a postgrad team comprising [Ari Anastassiou](https://www.linkedin.com/in/ari-anastassiou-6958a519/), [Matthew Aquilina](https://www.linkedin.com/in/matthewaq/), and [Adam DePauw](https://www.linkedin.com/in/adam-depauw/). This project was for our [Machine Learning Practical](http://www.inf.ed.ac.uk/teaching/courses/mlp/index-2018.html) module, during our AI Master's degree at The University of Edinburgh. Our team name is The Hidden Units, and our chosen task was titled _Forging the Perfect Recognition_.

Out of 127 participating teams, we were ranked third and were recognized by being awarded an IBM-sponsored prize for our work and achieving distinction.

## The Project
Many corporate companies have systems wherein employees can recognize or appraise fellow employees for their work. The aim of these recognitions is to increase employee engagement and foster a healthy working environment.

Our task was to develop a system that can improve the quality of these recognitions, such that they are impactful to their recipients. This is a novel task for which we found no previous work.

### Classifier
We experimented with a number of different text embeddings, such as `doc2vec`, as well as a number of transfer learning approaches, using external datasets. We implemented an exhaustive search through feature combinations and achieved a 76% classification accuracy when compared to domain-expert ratings, using a fully-connected deep architecture.

### Recommender
In order to recommend changes to a recognition, we use a generative approach that has as input a snippet of a recognition, and is trained to predict text as output that will improve the classification score of the recognition in question. Our baseline generative model is a character-level LSTM, which empirically produces well-formatted, but largely general and non-specific text. Our more advanced model, a Skip-Thoughts model, while producing less-legible text, has better semantic relationships to the input text. This  is useful for us, as we use this text to suggest *topics* for a user to include in the recognition to improve its rating.

## The Repo
Our repo is split into two major folders, `Code/` and `Outcomes`, which contain our codebase and our formal reports respectively. In `Code/`, an adventurer such as yourself will find `nnframework/`, `preprocessing/`, `'recommender/`, and `'skipthoughts/`. An empty `Data/` folder should be added alongside these as that is where our code is searching for the intended dataset to be used in this project.

### preprocessing/
This folder contains all our preprocessing steps, including part of speech analysis, `doc2vec` feature extraction, de-duplication implementations, amongst other useful little tricks.

### nnframework/
This folder has our classification neural network, along with inference mechanisms. We also have our LSTM implementation here.

### skipthoughts/
This folder contains the bulk of our generative approach. As explained in its own README file, but for completeness I'll describe it here too. We used the original author's codebase, but adapted it to work in Python 3 (round brackets everywhere -- yay!). We also included small adjustments such that the architecture works for our dataset, but the bulk of the code is as per the original paper, referenced later.

### recommender/
This folder contains our final recommender, which uses our generative models to recommend new *topics* based on the generation and domain-expert input. As a highlight, our method works! Details can be found in our report. :)

## Notes
Some data (and subsequent files) are unavailable in this repo due to sensitive data as per our original dataset providers. For any fearless soul who may want to reproduce parts of this work, and is struggling with any of these dependencies, please contact and of the authors of this work (our LinkedIn profiles are linked above), and we will gladly assist!

## The Legal Stuff
This project is released under the Modified BSD License.



