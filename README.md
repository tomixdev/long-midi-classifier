# Music Classification Based on Long-term Form

## Motivation 
A successful piece of music requires skillful ordering of diverse musical components. The recent development of sound synthesis technology and music generation algorithms has provided inspirational methods for generating components of music including melody, harmony, and timbre. Computational generation of an entire piece with a long-term musical narrative, however, has remained a challenge and a frequently neglected aspect of algorithmic music generation.

## What This Project Does
This project presents an experimental music classifier to classify two midi datasets: real classical compositions (labelled "good") and random 1-2-minute segments of them ("labelled "bad"). This labelling method is due to the belief that segmented pieces of music do not present long-term musical consistency while complete real compositions do.

## Ground Truth Example
The images below show midi clips with ground truth labels.
<br>
![Alt text](05-visualization/20230509_092423_ground_truth_labels.png)

## Prediction Example
The project built a model that classifies these images at approximately 85% accuracy. Blue labels in the picture below show wrong predictions.
<br>
![Alt text](05-visualization/20230509_092423_classification_result.png)

## How to use the model pretrained in this project?
* Pretrained models are available in `04-model-ckpt` folder. 
* The model currently assumes **performance-based** piano midi. 

## How to train your own model?
1. clone this repo
2. Put all midi files you want to classify as 'good' under `02-raw-midi-data/good` directory. 
3. Put all midi files you want to classify as 'bad' under `02-raw-midi-data/bad` directory.
4. Run `src/step2_2_midi2img_without_image_preprocessing.py` to convert midi files into images.
5. Run `src/step3_build_binary_classifier.py` to build a model. The model will be saved ujnder `04-model-ckpt/` directory.

## What to come next?
The work is in progress. Now that that a classifier based on the success of musical form is here, the next task is to built a music generator whose outcomes can convince the classifier of its musical consistency.