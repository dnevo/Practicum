Music Genre Classification with Python
======================================

A Guide to analysing Audio/Music signals in Python
--------------------------------------------------
[Dec 13,
2018](https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8?source=post_page-----c714d032f0d8----------------------)
· 9 min read

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/0_EfWqZYlS-GfMmqLj)

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/0_EfWqZYlS-GfMmqLj(1))

![](https://miro.medium.com/max/9184/0*EfWqZYlS-GfMmqLj)

Photo by
[Jean](https://unsplash.com/@dotgrid?utm_source=medium&utm_medium=referral)
on
[Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)

> Music is like a mirror, and it tells people a lot about who you are
> and what you care about, whether you like it or not. We love to say
> “you are what you stream,”:Spotify

[Spotify](https://en.wikipedia.org/wiki/Spotify), with a net worth of
[\$26
billion](https://www.cnbc.com/2018/04/04/spotify-chiefs-net-worth-tops-2-billion-after-nyse-debut.html)
is reigning the [music streaming
platform](https://en.wikipedia.org/wiki/Comparison_of_on-demand_streaming_music_services)
today. It currently has millions of songs in its database and claims to
have the right music score for everyone. **Spotify’s Discover Weekly**
service has become a hit with the millennials. Needless to say, Spotify
has invested a lot in research to improve the way users find and listen
to music. Machine Learning is at the
[core](https://www.thestar.com/entertainment/2016/01/14/meet-the-man-classifying-every-genre-of-music-on-spotify-all-1387-of-them.html)
of their research. From NLP to Collaborative filtering to Deep Learning,
Spotify uses them all. Songs are analyzed based on their digital
signatures for some factors, including tempo, acoustics, energy,
danceability etc. to answer that impossible old first-date query: **What
kind of music are you into?**

* * * * *

Objective {#9438 .kx .ky .bj .bi .kz .la .lb .lc .ld .le .lf .lg .lh .li .lj .lk .ll .lm .ln .lo .lp .ct data-selectable-paragraph=""}
=========

[Companies](http://cs229.stanford.edu/proj2016/report/BurlinCremeLenain-MusicGenreClassification-report.pdf)nowadays
use music classification, either to be able to place recommendations to
their customers (such as Spotify, Soundcloud) or simply as a product
(for example Shazam). Determining music genres is the first step in that
direction. Machine Learning techniques have proved to be quite
successful in extracting trends and patterns from the large pool of
data. The same principles are applied in Music Analysis also.

*In this article, we shall study how to analyse an audio/music signal in
Python. We shall then utilise the skills learnt to****classify music
clips into different genres.***

Audio Processing with Python {#f019 .kx .ky .bj .bi .kz .la .mb .lc .ld .mc .lf .lg .md .li .lj .me .ll .lm .mf .lo .lp .ct data-selectable-paragraph=""}
============================

Sound is represented in the form of an **audio** signal having
parameters such as frequency, bandwidth, decibel etc. A typical audio
signal can be expressed as a function of Amplitude and Time.

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_akRbhl8739UEDuKHkOUR1Q.png)

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_akRbhl8739UEDuKHkOUR1Q(1).png)

![](https://miro.medium.com/max/1978/1*akRbhl8739UEDuKHkOUR1Q.png)

[source](https://docs.google.com/presentation/d/1zzgNu_HbKL2iPkHS8-qhtDV20QfWt9lC3ZwPVZo8Rw0/pub?start=false&loop=false&delayms=3000&slide=id.g5a7a9806e_0_84)

These sounds are available in many formats which makes it possible for
the computer to read and analyse them. Some examples are:

-   **mp3 format**
-   **WMA (Windows Media Audio) format**
-   **wav (Waveform Audio File) format**

Audio Libraries {#1a0c .kx .ky .bj .bi .kz .la .mb .lc .ld .mc .lf .lg .md .li .lj .me .ll .lm .mf .lo .lp .ct data-selectable-paragraph=""}
===============

Python has some great [libraries](https://wiki.python.org/moin/Audio/)
for audio processing like Librosa and PyAudio.There are also built-in
modules for some basic audio functionalities.

We will mainly use two libraries for audio acquisition and playback:

1. Librosa {#b3e7 .mq .ky .bj .bi .kz .mr .ms .ke .mt .mu .kg .mv .mw .go .mx .my .gr .mz .na .gu .nb .ct data-selectable-paragraph=""}
----------

It is a Python module to analyze audio signals in general but geared
more towards music. It includes the nuts and bolts to build a MIR(Music
information retrieval) system. It has been very well
[documented](https://librosa.github.io/librosa/) along with a lot of
examples and tutorials.

*For a more advanced introduction which describes the package design
principles, please refer to the*[*librosa
paper*](http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf)*at*[*SciPy
2015*](http://scipy2015.scipy.org/)*.*

**Installation**

``` {.im .in .io .ip .iq .nc .nd .dz}
pip install librosaorconda install -c conda-forge librosa
```

To fuel more audio-decoding power, you can install *ffmpeg* which ships
with many audio decoders.

2. IPython.display.Audio {#1edb .mq .ky .bj .bi .kz .mr .ms .ke .mt .mu .kg .mv .mw .go .mx .my .gr .mz .na .gu .nb .ct data-selectable-paragraph=""}
------------------------

`IPython.display.Audio`{.jb .ni .nj .nk .ne .b} lets you play audio
directly in a jupyter notebook.

Loading an audio file {#135e .kx .ky .bj .bi .kz .la .mb .lc .ld .mc .lf .lg .md .li .lj .me .ll .lm .mf .lo .lp .ct data-selectable-paragraph=""}
=====================

``` {.im .in .io .ip .iq .nc .nd .dz}
import librosaaudio_path = '../T08-violin.wav'x , sr = librosa.load(audio_path)print(type(x), type(sr))<class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)(396688,) 22050
```

This returns an audio time series as a numpy array with a default
sampling rate(sr) of 22KHZ mono. We can change this behaviour by saying:

``` {.im .in .io .ip .iq .nc .nd .dz}
librosa.load(audio_path, sr=44100)
```

to resample at 44.1KHz, or

``` {.im .in .io .ip .iq .nc .nd .dz}
librosa.load(audio_path, sr=None)
```

to disable resampling.

The sample**rate** is the number of samples of audio carried per second,
measured in Hz or kHz.

Playing Audio {#a8fd .kx .ky .bj .bi .kz .la .mb .lc .ld .mc .lf .lg .md .li .lj .me .ll .lm .mf .lo .lp .ct data-selectable-paragraph=""}
=============

Using,`IPython.display.Audio`{.jb .ni .nj .nk .ne .b} to play the audio

``` {.im .in .io .ip .iq .nc .nd .dz}
import IPython.display as ipdipd.Audio(audio_path)
```

This returns an audio widget in the jupyter notebook as follows:

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_ktoXTt51zFSTMgiuv5ZCgg.png)

![](https://miro.medium.com/max/684/1*ktoXTt51zFSTMgiuv5ZCgg.png)

screenshot of the Ipython audio widget

This widget won’t work here, but it will work in your notebooks. I have
uploaded the same to SoundCloud so that we can listen to it.

You can even use an mp3 or a WMA format for the audio example.

Visualizing Audio {#a642 .kx .ky .bj .bi .kz .la .mb .lc .ld .mc .lf .lg .md .li .lj .me .ll .lm .mf .lo .lp .ct data-selectable-paragraph=""}
=================

Waveform {#3660 .mq .ky .bj .bi .kz .mr .ms .ke .mt .mu .kg .mv .mw .go .mx .my .gr .mz .na .gu .nb .ct data-selectable-paragraph=""}
--------

We can plot the audio array using `librosa.display.waveplot`{.jb .ni .nj
.nk .ne .b}:

``` {.im .in .io .ip .iq .nc .nd .dz}
%matplotlib inlineimport matplotlib.pyplot as pltimport librosa.displayplt.figure(figsize=(14, 5))librosa.display.waveplot(x, sr=sr)
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_tspcooMfQnmYPwxMKWhTmg.png)

![](https://miro.medium.com/max/1414/1*tspcooMfQnmYPwxMKWhTmg.png)

Here, we have the plot of the amplitude envelope of a waveform.

Spectrogram {#6405 .mq .ky .bj .bi .kz .mr .ms .ke .mt .mu .kg .mv .mw .go .mx .my .gr .mz .na .gu .nb .ct data-selectable-paragraph=""}
-----------

A [**spectrogram**](https://en.wikipedia.org/wiki/Spectrogram) is a
visual representation of the
[spectrum](https://en.wikipedia.org/wiki/Spectral_density) of
[frequencies](https://en.wikipedia.org/wiki/Frequencies) of
[sound](https://en.wikipedia.org/wiki/Sound) or other signals as they
vary with time. Spectrograms are sometimes called **sonographs**,
**voiceprints**, or **voicegrams**. When the data is represented in a 3D
plot, they may be called **waterfalls**. In 2-dimensional arrays, the
first axis is frequency while the second axis is time.

We can display a spectrogram using. `librosa.display.specshow.`{.jb .ni
.nj .nk .ne .b}

``` {.im .in .io .ip .iq .nc .nd .dz}
X = librosa.stft(x)Xdb = librosa.amplitude_to_db(abs(X))plt.figure(figsize=(14, 5))librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')plt.colorbar()
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_7jZRCE58z2QTulOIIHw_LA.png)

![](https://miro.medium.com/max/1402/1*7jZRCE58z2QTulOIIHw_LA.png)

The vertical axis shows frequencies (from 0 to 10kHz), and the
horizontal axis shows the time of the clip. Since we see that all action
is taking place at the bottom of the spectrum, we can convert the
frequency axis to a logarithmic one.

``` {.im .in .io .ip .iq .nc .nd .dz}
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')plt.colorbar()
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_JttzQLEFRzAb3clp7dkfKg.png)

![](https://miro.medium.com/max/946/1*JttzQLEFRzAb3clp7dkfKg.png)

Writing Audio {#d805 .kx .ky .bj .bi .kz .la .mb .lc .ld .mc .lf .lg .md .li .lj .me .ll .lm .mf .lo .lp .ct data-selectable-paragraph=""}
=============

`librosa.output.write_wav`{.jb .ni .nj .nk .ne .b} saves a NumPy array
to a WAV file.

``` {.im .in .io .ip .iq .nc .nd .dz}
librosa.output.write_wav('example.wav', x, sr)
```

Creating an audio signal {#8bd6 .kx .ky .bj .bi .kz .la .mb .lc .ld .mc .lf .lg .md .li .lj .me .ll .lm .mf .lo .lp .ct data-selectable-paragraph=""}
========================

Let us now create an audio signal at 220Hz. An audio signal is a numpy
array, so we shall create one and pass it into the audio function.

``` {.im .in .io .ip .iq .nc .nd .dz}
import numpy as npsr = 22050 # sample rateT = 5.0    # secondst = np.linspace(0, T, int(T*sr), endpoint=False) # time variablex = 0.5*np.sin(2*np.pi*220*t)# pure sine wave at 220 HzPlaying the audioipd.Audio(x, rate=sr) # load a NumPy arraySaving the audiolibrosa.output.write_wav('tone_220.wav', x, sr)
```

So, here it is- first sound signal created by you.🙌

Feature extraction {#aa21 .kx .ky .bj .bi .kz .la .mb .lc .ld .mc .lf .lg .md .li .lj .me .ll .lm .mf .lo .lp .ct data-selectable-paragraph=""}
==================

Every audio signal consists of many features. However, we must extract
the characteristics that are relevant to the problem we are trying to
solve. The process of extracting features to use them for analysis is
called feature extraction. Let us study about few of the features in
detail.

-   **Zero Crossing Rate**

The [zero crossing
rate](https://en.wikipedia.org/wiki/Zero-crossing_rate) is the rate of
sign-changes along a signal, i.e., the rate at which the signal changes
from positive to negative or back. This feature has been used heavily in
both [speech
recognition](https://en.wikipedia.org/wiki/Speech_recognition) and
[music information
retrieval](https://en.wikipedia.org/wiki/Music_information_retrieval).
It usually has higher values for highly percussive sounds like those in
metal and rock.

Let us calculate the**zero crossing rate**for our example audio clip.

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_Re6tj77KWLypP3gY7MaXjA.png)

![](https://miro.medium.com/max/1356/1*Re6tj77KWLypP3gY7MaXjA.png)

``` {.im .in .io .ip .iq .nc .nd .dz}
# Load the signalx, sr = librosa.load('../T08-violin.wav')#Plot the signal:plt.figure(figsize=(14, 5))librosa.display.waveplot(x, sr=sr)
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_m3v38dqyykefBp0OiNRBPg.png)

![](https://miro.medium.com/max/1396/1*m3v38dqyykefBp0OiNRBPg.png)

``` {.im .in .io .ip .iq .nc .nd .dz}
# Zooming inn0 = 9000n1 = 9100plt.figure(figsize=(14, 5))plt.plot(x[n0:n1])plt.grid()
```

There appear to be 6 zero crossings. Let’s verify with librosa.

``` {.im .in .io .ip .iq .nc .nd .dz}
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)print(sum(zero_crossings))6
```

-   [**Spectral
    Centroid**](https://en.wikipedia.org/wiki/Spectral_centroid)

It indicates where the ”centre of mass” for a sound is located and is
calculated as the weighted mean of the frequencies present in the sound.
Consider two songs, one from a blues genre and the other belonging to
metal. Now as compared to the blues genre song which is the same
throughout its length, the metal song has more frequencies towards the
end. **So spectral centroid for blues song will lie somewhere near the
middle of its spectrum while that for a metal song would be towards its
end.**

`librosa.feature.spectral_centroid`{.jb .ni .nj .nk .ne .b} computes the
spectral centroid for each frame in a signal:

``` {.im .in .io .ip .iq .nc .nd .dz}
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]spectral_centroids.shape(775,)# Computing the time variable for visualizationframes = range(len(spectral_centroids))t = librosa.frames_to_time(frames)# Normalising the spectral centroid for visualisationdef normalize(x, axis=0):    return sklearn.preprocessing.minmax_scale(x, axis=axis)#Plotting the Spectral Centroid along the waveformlibrosa.display.waveplot(x, sr=sr, alpha=0.4)plt.plot(t, normalize(spectral_centroids), color='r')
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_1Vu8GXxY_CmMBLmcypEjvw.png)

![](https://miro.medium.com/max/892/1*1Vu8GXxY_CmMBLmcypEjvw.png)

There is a rise in the spectral centroid towards the end.

-   **Spectral Rolloff**

It is a measure of the shape of the signal. It represents the frequency
below which a specified percentage of the total spectral energy, e.g.
85%, lies.

`librosa.feature.spectral_rolloff`{.jb .ni .nj .nk .ne .b} computes the
rolloff frequency for each frame in a signal:

``` {.im .in .io .ip .iq .nc .nd .dz}
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]librosa.display.waveplot(x, sr=sr, alpha=0.4)plt.plot(t, normalize(spectral_rolloff), color='r')
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_pOp0miIvDc1wxvS1iwrwCQ.png)

![](https://miro.medium.com/max/944/1*pOp0miIvDc1wxvS1iwrwCQ.png)

-   [**Mel-Frequency Cepstral
    Coefficients**](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

The Mel frequency cepstral coefficients (MFCCs) of a signal are a small
set of features (usually about 10–20) which concisely describe the
overall shape of a spectral envelope. It models the characteristics of
the human voice.

Let’ work with a simple loop wave this time.

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_kLln6ejJJsTDn9B0zLALUA.png)

![](https://miro.medium.com/max/882/1*kLln6ejJJsTDn9B0zLALUA.png)

``` {.im .in .io .ip .iq .nc .nd .dz}
x, fs = librosa.load('../simple_loop.wav')librosa.display.waveplot(x, sr=sr)
```

`librosa.feature.mfcc`{.jb .ni .nj .nk .ne .b} computes MFCCs across an
audio signal:

``` {.im .in .io .ip .iq .nc .nd .dz}
mfccs = librosa.feature.mfcc(x, sr=fs)print mfccs.shape(20, 97)#Displaying  the MFCCs:librosa.display.specshow(mfccs, sr=sr, x_axis='time')
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_a6OcRvqCNDEix02CI5dK6w.png)

![](https://miro.medium.com/max/770/1*a6OcRvqCNDEix02CI5dK6w.png)

Here mfcc computed 20 MFCC s over 97 frames.

We can also perform feature scaling such that each coefficient dimension
has zero mean and unit variance:

``` {.im .in .io .ip .iq .nc .nd .dz}
import sklearnmfccs = sklearn.preprocessing.scale(mfccs, axis=1)print(mfccs.mean(axis=1))print(mfccs.var(axis=1))librosa.display.specshow(mfccs, sr=sr, x_axis='time')
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_6ar_u2SROhmOnvgimnI48Q.png)

![](https://miro.medium.com/max/802/1*6ar_u2SROhmOnvgimnI48Q.png)

-   [**Chroma
    Frequencies**](https://labrosa.ee.columbia.edu/matlab/chroma-ansyn/)

Chroma features are an interesting and powerful representation for music
audio in which the entire spectrum is projected onto 12 bins
representing the 12 distinct semitones (or chroma) of the musical
octave.

`librosa.feature.chroma_stft `{.jb .ni .nj .nk .ne .b}is used for
computation

``` {.im .in .io .ip .iq .nc .nd .dz}
# Loadign the filex, sr = librosa.load('../simple_piano.wav')hop_length = 512chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)plt.figure(figsize=(15, 5))librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
```

![](./Music%20Genre%20Classification%20with%20Python%20-%20Towards%20Data%20Science_files/1_31ZW3GzKofk4jsSyRMwubg.png)

![](https://miro.medium.com/max/1352/1*31ZW3GzKofk4jsSyRMwubg.png)

* * * * *

Case Study: Classify songs into different genres. {#2093 .kx .ky .bj .bi .kz .la .lb .lc .ld .le .lf .lg .lh .li .lj .lk .ll .lm .ln .lo .lp .ct data-selectable-paragraph=""}
=================================================

*After having an overview of the acoustic signal, their features and
their feature extraction process, it is time to utilise our newly
developed skill to work on a Machine Learning Problem.*

Objective {#b227 .mq .ky .bj .bi .kz .mr .ms .ke .mt .mu .kg .mv .mw .go .mx .my .gr .mz .na .gu .nb .ct data-selectable-paragraph=""}
---------

In his section, we will try to model a classifier to classify songs into
different genres. Let us assume a scenario in which, for some reason, we
find a bunch of randomly named MP3 files on our hard disk, which are
assumed to contain music. Our task is to sort them according to the
music genre into different folders such as jazz, classical, country,
pop, rock, and metal.

Dataset {#1a43 .mq .ky .bj .bi .kz .mr .ms .ke .mt .mu .kg .mv .mw .go .mx .my .gr .mz .na .gu .nb .ct data-selectable-paragraph=""}
-------

We will be using the famous
[GITZAN](http://marsyasweb.appspot.com/download/data_sets/) dataset for
our case study. This dataset was used for the well-known paper in genre
classification “ [Musical genre classification of audio
signals](https://ieeexplore.ieee.org/document/1021072)“ by G. Tzanetakis
and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

The dataset consists of 1000 audio tracks each 30 seconds long. It
contains 10 genres namely, **blues, classical, country, disco, hiphop,
jazz, reggae, rock, metal and pop.**Each genre consists of 100 sound
clips.

Preprocessing the Data {#1482 .mq .ky .bj .bi .kz .mr .ms .ke .mt .mu .kg .mv .mw .go .mx .my .gr .mz .na .gu .nb .ct data-selectable-paragraph=""}
----------------------

Before training the classification model, we have to transform raw data
from audio samples into more meaningful representations. The audio clips
need to be converted from .au format to .wav format to make it
compatible with python’s wave module for reading audio files. I used the
open source [SoX](http://sox.sourceforge.net/)module for the conversion.
Here is a handy [cheat
sheet](https://www.stefaanlippens.net/audio_conversion_cheat_sheet/) for
SoX conversion.

``` {.im .in .io .ip .iq .nc .nd .dz}
sox input.au output.wav
```

Classification {#e9a0 .mq .ky .bj .bi .kz .mr .ms .ke .mt .mu .kg .mv .mw .go .mx .my .gr .mz .na .gu .nb .ct data-selectable-paragraph=""}
--------------

-   **Feature Extraction**

We then need to extract meaningful features from audio files. To
classify our audio clips, we will choose 5 features, i.e. Mel-Frequency
Cepstral Coefficients, Spectral Centroid, Zero Crossing Rate, Chroma
Frequencies, Spectral Roll-off. All the features are then appended into
a .csv file so that classification algorithms can be used.

-   **Classification**

Once the features have been extracted, we can use existing
classification algorithms to classify the songs into different genres.
You can either use the spectrogram images directly for classification or
can extract the features and use the classification models on them.

Either way, a lot of experimentation can be done in terms of models. You
are free to experiment and improve your results. Using a CNN model (on
the spectrogram images) gives a better accuracy and its worth a try.

Next Steps
==========

Music Genre Classification is one of the many branches of [Music
Information
Retrieval](https://en.wikipedia.org/wiki/Music_information_retrieval).
From here you can perform other tasks on musical data like beat
tracking, music generation, recommender systems, track separation and
instrument recognition etc. Music analysis is a diverse field and also
an interesting one. A music session somehow represents a moment for the
user. Finding these moments and describing them is an interesting
challenge in the field of Data Science.
