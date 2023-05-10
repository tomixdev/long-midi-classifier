# Music Classification Based on Long-term Form


# Motivation
* instrumental music requires long-term consistency. Yet, this aspect is often ignored in music generation study, which tends to focus on local music generation (short term). 

* A successful piece of music requires skillful ordering of diverse musical components. The recent development of sound synthesis technology and music generation algorithms has provided inspirational methods for generating components of music including melody, harmony, and timbre. Computational generation of an entire piece with a long-term musical narrative, however, has remained a challenge. My research proposes the concept and implementation of "Form Sampling"—the third type of sampling technology after sampling of sound (i.e. conventional sampling) and sampling of space (i.e., convolution reverb). The investigated algorithm can: (1) search for an audio segment from an audio database; (2) automatically map its appropriate features to parameters of an existing short-term music generation algorithm; (3) and generate a "vomit draft" of an entire piece that embrace a global design and leaves the comfort of editing to a music creator. Both qualitative and quantitative results of a randomized blind test conducted for 23 music composition domain experts suggested the effectiveness of the algorithm for long-term musical form generation. I anticipate my study to be a starting point for more sophisticated end-to-end models for compositional creativity in music. Furthermore, designing an algorithm for the global form designing process in music creation will be critical to make AI-assisted music composition technologies practical and meaningful to a wider range of real-world music creators. 


![Alt text](05-visualization/20230509_092423_ground_truth_labels.png)

![Alt text](05-visualization/20230509_092423_classification_result.png)



# How to use pretrained model?
* pretrained models are available. to use use 
```python
import torch
torch.

```

# How to train your own model?
* only for piano midi. prefarably performance midi
* `git clone`
* `put raw-midi-data`, if place any midi files that you want to classifiy as "good" to the "good" folder. Place anything you want to classify as "bad" to "bad" folder.
* run src step2_2_midi2img_without_image_preprocessing.py to convert midi files into 



# Future work
The work is in progress. The final goal is to create a music generator that can generate instrumental piano music with a formal consistency. 

* Create a generator  that can be classified as "good"
