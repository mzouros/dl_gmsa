# Project: Greek Music Sentiment Analysis

This repo contains a single Jupyter Notebook (Python) for the Deep Learning Course at the MSc Program of AI 2020-2022, organized by NCSR Demokritos and University of Piraeus.

Instructor: Mr. Theodoros Giannakopoulos - [tygiannak](https://github.com/tyiannak)

## Notebook Sections

### Imports

* [matplotlib](https://github.com/matplotlib/matplotlib), [plotly](https://github.com/plotly) and [seaborn](https://github.com/mwaskom/seaborn) for Visualization

* [numpy](https://github.com/numpy/numpy) for Matrices

* [opencv](https://github.com/opencv/opencv) for Computer Vision

* [pydub](https://github.com/jiaaro/pydub) and [scipy](https://github.com/scipy/scipy) for .wav conversion and manipulation

* [librosa](https://github.com/librosa/librosa) for Music Analysis

* [sklearn](https://github.com/scikit-learn/scikit-learn) for Machine Learning

* [keras](https://github.com/keras-team/keras) for Deep Learning

### Data Collection

* Approximately 10k Greek songs were collected spanning from the Interwar Period (1920-1940) and Greek Junta / Dictatorship (1967-1974), until today (2021)
* The songs collected were in various forms (.mp3, .wma, .wav) and their aggregated size exceeds 120 GB
* Initial Dataset: https://raw.githubusercontent.com/mzouros/dl_gmsa/main/Greek_Songs_List.txt

### Data Preparation

* Data cleanup:
  * Remove any unnecessary files (.txt, .jpg, .zip, etc)
  * Delete songs with generic names (eg. Track01)
  * Delete duplicate songs
  * Try and narrow down live performance songs
* Data matching:
  * Create folders for each Singer OR Songwriter OR Lyricist OR Producer (according to Spotify's registries) and assort each song to a single specific folder (locally)
  * Create the same folders on Spotify, then search which of the songs in our DB is a registry on Spotify. For each matching, insert the song to its corresponding folder
  * 1 by 1 matching each song to its Spotify counterpart. Rename each song of the dataset according to Spotify's Track name registries 

### Data Validation

*  Deal with bad matching / mispellings
*  Validate Playlist names (Spotify) against Album names (locally)
*  Validate Playlist track names (Spotify) against Album track names (locally)

### Feature Extraction

* Feature extraction via Spotify's API and export data to .csv
* Some of the features extracted were Danceability, Energy, Valence, Liveness and Loudness
* Song Features: https://github.com/mzouros/dl_gmsa/blob/main/Spotify_Tracks.csv 

### Data Preprocessing

* Transform .mp3 and .wma files to .wav
* Resample all tracks to 8k Hz and monophonic (mono) sound
* Create songs' sentiments classification logic, according to Arousal-Valence Model
* Classify each song to a specific sentiment according to its values from the .csv file

![alt text](https://i.imgur.com/hJtuElb.png)
![alt text](https://i.imgur.com/R77woaW.png)

### Data Exploration

* Optical
  * Waveforms
  * Spectograms
  * Mel Spectograms & Laplacian Mel Spectograms
  * Chromas
  * Tonnetz
  * CENS
* Features
  * MFCCs
  * STFTs
  * ZCRs

![alt text](https://i.imgur.com/FUpXhn5.png)
![alt text](https://i.imgur.com/c73Cfzi.png)
![alt text](https://i.imgur.com/3JEIVQH.png)
![alt text](https://i.imgur.com/vCNPqwQ.png)
![alt text](https://i.imgur.com/o3ImYN7.png)
![alt text](https://i.imgur.com/ZW0sZm4.png)

### Data Revision and Augmentation

After experimenting with the initial dataset:

* Consider more data (via SpecAugmentation - Frequency Mask)
* Reconsider total of images per sentiment category (500 images for each category - 4k images), so to have the perfect balanced set
* Reconsider image size from 128x128 to 256x256

![alt text](https://i.imgur.com/8K2gsmm.png)

### CNN Implementation and Results

* Lots of different architectures have been tested, resulting on similar results (<0.4 validation accuracy)
* In all the architectures our model seems to not be able to learn after a while
* Tried most of overfitting avoidance techniques (kernel regularization, batch normalization, dropout, ES)
* Tried with different model sizes (layers, nodes), batch sizes, kernel & stride sizes, dropout values, number of epochs
* Tried with a perfect balanced set of 500 samples for each sentiment (4k samples total) - after data augmentation

![alt text](https://i.imgur.com/vu9LkE8.png)![alt text](https://i.imgur.com/vZicloM.png)![alt text](https://i.imgur.com/BaNFwhL.png)

### Discussion

* Problems during implementation:
  * 1 by 1 matching very time consuming
  * Lots of duplicates required extra work (eg Stavros Xarxakos and Stavros Ksarhakos, both indicating the same Artist)
  * Spotify classifies as Artists singers, songwriters, lyricists and producers. Need to take all of those into consideration when searching if a specific song in our dataset exists as a registry in Spotify
  * Annotating an audio recording is challenging. How many emotions should we define to recognize?
  * Emotions are subjective, people would interpret it differently. It is hard to define the notion of emotions.
  * Audio Analysis through image classification tend to have bad accuracy results (<0.5) (https://towardsdatascience.com/whats-wrong-with-spectrograms-and-cnns-for-audio-processing-311377d7ccd)
* Food for thought:
  * How Spotify quantify its track's variables is not clear, but usually in fuzzy logic, a team of specialists define the pertinence degree between qualitative aspects
  * A happy song may have sad lyrics and vice versa. A perfect song sentiment analysis would require both text and sound analysis and classification.  

### Future Work

Experiment with:
* Bigger Dataset (10k+ images)
* Bigger image size (maybe 512x512)
* Pre-trained Classifiers
* Classification in 4 sentiments (Happy, Calm, Sad, Angry) instead of 8
* Different NN architectures (eg CNN-RNN in parallel)
* Instead of feature extraction through Spotify's API, use an Annotation App, so volunteers individuals can annotate according to their belief (eg AnnoEmo app)

## Authors

* **Michael Zouros** - [mzouros](https://github.com/mzouros)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Theodoros Giannakopoulos - [tygiannak](https://github.com/tyiannak)

