# Genre_Pred
This project was done in collaboration with Jeffrey Kuo, Bryan Wang, and Tessa Weiss at the University of California, Berkeley for Stat 254. Please note that the following is a simplified overview, and further information can be found in [Method_Details.pdf](https://github.com/Ttantivi/genre_pred/blob/main/Method_Details.pdf)

Thie project builds upon the work of [fma](https://github.com/mdeff/fma) and [torchvggish](https://github.com/harritaylor/torchvggish).

## Central Goal and Project Introduction
The goal of our project was to reproduce state-of-the-art results in genre classification by utilizing convolutional neural networks (CNNs). We compared two different techniques to see which performed better: using a pretrained CNN (VGGish) as an embedding model for other classification models (SVM + XGBoost), and finetuning the pretrained CNN with additional layers for classifications.

### Motivation
Automatic genre prediction is an important problem in the music information retrieval (MIR) space. Having an accurate genre prediction model is useful for building music recommender systems. On platforms such as Spotify and Apple Music, being able to accurately predict the genres that a user listens to allows us to recommend songs of the same type to the user. 

However, this problem is complicated for two main reasons. The first is that a song can fit multiple genres, and the classification of a song can be subjective. The second is that there can be high variability within a genre. For example, the "Electronic" music genre contains high BPM party songs with straightforward beats, as well as more avante-garde, LoFi selections with extremely complex musical structure.

To mitigate these problems as much as possible, we decided to restrict the scope of our project to only predicting eight very distinct genres. We used the "small" partition of the FMA dataset, which contains exactly 8,000 songs. The genres present in this dataset are: "Hip-Hop", "Pop", "Folk", "Experimental", "Rock", "International", "Electronic", and "Instrumental".

## Notebook Table of Contents
* [baseline_tim.ipynb](https://github.com/Ttantivi/genre_pred/blob/main/Notebooks/baseline_tim.ipynb): Reproduces FMA's baseline results.
* [torch_transfer.ipynb](https://github.com/Ttantivi/genre_pred/blob/main/Notebooks/torch_transfer.ipynb): Uses CNN as feature extractor and runs traditional ML algorithms for prediction.
* [torch_transfer_GPU.ipynb](https://github.com/Ttantivi/genre_pred/blob/main/Notebooks/torch_transfer_GPU.ipynb): Uses finetuned CNN as classifier for prediction.