# Handwriting Recognition Using Im2Latex

This repository uses the architecture proposed in "What You Get Is What You See: A Visual Markup Decompiler" (http://arxiv.org/pdf/1609.04938v1.pdf) to the problem of Handwriting Recognition. The base implementation was done in Tensorflow by ritheshkumar95/im2latex-tensorflow (forked) and was modified to work for Handwriting Recognition. The original Torch implementation of the paper is located here: https://github.com/harvardnlp/im2markup/blob/master/

    What You Get Is What You See: A Visual Markup Decompiler  
    Yuntian Deng, Anssi Kanervisto, and Alexander M. Rush
    http://arxiv.org/pdf/1609.04938v1.pdf

This deep learning framework can be used to learn a representation of an image. In this case, our input image is an image of text and we are converting this image to an ASCII representation.

<p align="center"><img src="http://lstm.seas.harvard.edu/latex/network.png" width="400"></p>

Below is an example of an input image of text:

![alt text](https://raw.githubusercontent.com/nexusapoorvacus/Handwriting-Recognition/master/images/data/a01/a01-000u/a01-000u-00-01.png)

The goal is to infer the following ASCII text:

```
MOVE
```

## Prerequsites

Most of the code is written in tensorflow, with Python for preprocessing.

### Preprocess
The proprocessing for this dataset is exactly reproduced as the original torch implementation by the HarvardNLP group

Python

* Pillow
* numpy

Optional: We use Node.js and KaTeX for preprocessing [Installation](https://nodejs.org/en/)

##### pdflatex [Installaton](https://www.tug.org/texlive/)

Pdflatex is used for rendering LaTex during evaluation.

##### ImageMagick convert [Installation](http://www.imagemagick.org/script/index.php)

Convert is used for rending LaTex during evaluation.

##### Webkit2png [Installation](http://www.paulhammond.org/webkit2png/)

Webkit2png is used for rendering HTML during evaluation.

### Preprocessing Instructions

The images in the dataset contain a LaTeX formula rendered on a full page. To accelerate training, we need to preprocess the images.

Please download the training data from https://zenodo.org/record/56198#.WFojcXV94jA and extract into source (master) folder.

```
cd im2markup
```

```
python scripts/preprocessing/preprocess_images.py --input-dir ../formula_images --output-dir ../images_processed
```

The above command will crop the formula area, and group images of similar sizes to facilitate batching.

Next, the LaTeX formulas need to be tokenized or normalized.

```
python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file ../im2latex_formulas.lst --output-file formulas.norm.lst
```

The above command will normalize the formulas. Note that this command will produce some error messages since some formulas cannot be parsed by the KaTeX parser.

Then we need to prepare train, validation and test files. We will exclude large images from training and validation set, and we also ignore formulas with too many tokens or formulas with grammar errors.

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir ../images_processed --label-path formulas.norm.lst --data-path ../im2latex_train.lst --output-path train.lst
```

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir ../images_processed --label-path formulas.norm.lst --data-path ../im2latex_validate.lst --output-path validate.lst
```

```
python scripts/preprocessing/preprocess_filter.py --no-filter --image-dir ../images_processed --label-path formulas.norm.lst --data-path ../im2latex_test.lst --output-path test.lst
```

Finally, we generate the vocabulary from training set. All tokens occuring less than (including) 1 time will be excluded from the vocabulary.

```
python scripts/preprocessing/generate_latex_vocab.py --data-path train.lst --label-path formulas.norm.lst --output-file latex_vocab.txt
```

Train, Test and Valid images need to be segmented into buckets based on image size (height, width) to facilitate batch processing.

train_buckets.npy, valid_buckets.npy, test_buckets.npy can be generated using the DataProcessing.ipynb script

```
### Run the individual cells from this notebook
ipython notebook DataProcessing.ipynb
```

## Train

```
python attention.py
```
Default hyperparameters used:
* BATCH_SIZE      = 20
* EMB_DIM         = 80
* ENC_DIM         = 256
* DEC_DIM         = ENC_DIM*2
* D               = 512 (#channels in feature grid)
* V               = 502 (vocab size)
* NB_EPOCHS       = 50
* H               = 20  (Maximum height of feature grid)
* W               = 50  (Maximum width of feature grid)

The train NLL drops to 0.08 after 18 epochs of training on 24GB Nvidia M40 GPU.

## Test

predict() function in the attention.py script can be called to predict from validation or test sets.

Predict.ipynb script displays and renders the results saved by the predict() function

## Evaluate

attention.py scores the train set and validation set after each epoch (measures mean train NLL, perplexity)

#### Scores from this implementation

![results_1](results_1.png)
![results_2](results_2.png)

## Weight files
[Google Drive](https://drive.google.com/drive/folders/0BwbIUfIM1M8sc0tEMGk1NGlKZTA?usp=sharing)

## Visualizing the attention mechanism

![att_1](Pictures/Attention_1.png)

![att_2](Pictures/Attention_2.png)

![att_3](Pictures/Attention_3.png)

![att_4](Pictures/Attention_4.png)

![att_5](Pictures/Attention_5.png)

![att_6](Pictures/Attention_6.png)

![att_7](Pictures/Attention_7.png)

![att_8](Pictures/Attention_8.png)

![att_9](Pictures/Attention_9.png)

![att_10](Pictures/Attention_10.png)

![att_11](Pictures/Attention_11.png)
