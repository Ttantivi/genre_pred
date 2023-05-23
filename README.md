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
* [utils.py](https://github.com/Ttantivi/genre_pred/blob/main/utils.py): Helper functions from [fma](https://github.com/mdeff/fma).

To reproduce our results, download the MP3 files from [fma](https://github.com/mdeff/fma), named *fma_small.zip*.

## How CNNs and Mel-Spectrogram Work
CNNs are a class of deep neural networks that have been successfully applied to various computer vision tasks. CNNs are designed to automatically extract and learn spatial features from images by using convolutional layers, pooling layers, and fully connected layers. In a CNN, the convolutional layers apply a set of learnable filters to the input image, which detect different features such as edges, corners, and textures. The pooling layers downsample the feature maps by reducing their spatial resolution while retaining the most important information. The fully connected layers then process the high-level features and produce the final output of the network. By stacking multiple convolutional layers, pooling layers, and fully connected layers, a CNN can learn increasingly complex representations of the input image, which makes it suitable for a wide range of visual recognition tasks.

The reason why CNNs are so successful in music genre classification is because the audio recognition task is converted into a visual recognition task by utilizing mel-spectrograms. 

A mel-spectrogram is essentially an image that is created from an audio file, which is is a one-dimensional array of floats. Each float represents a sample of the song at a particular time point. Every audio file also specifies a sample rate, which is the number of samples per second of audio. For example, Spotify uses mp3 files with a 44.1 kHz sample rate and 16-bit float precision. This means that there are 44,100 floats (in 16-bit precision) per second of audio in the song.
    
Thus, even 30 seconds of audio will constitute an array of about 1.3 million floats. This input is too high-dimensional for most conventional models to handle, including feedforward neural networks. Even models built to handle time dependency, such as transformers or recurrent neural networks, will perform sub-optimally because of the sheer length that possible dependencies can take. Figure 1 shows an example of a mel-spectrogram.

![mel_spectrogram_ex](./Images/mel_spectrogram_ex.png)