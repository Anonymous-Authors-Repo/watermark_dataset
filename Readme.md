# Did You Train on My Dataset? Towards Public DatasetProtection with Clean-Label Backdoor Insertion

This is the pytorch implemention for paper **“Did You Train on My Dataset? Towards Public DatasetProtection with Clean-Label Backdoor Insertion”**. The huge supporting training data on the Internet has been a key factor in the success of deep learning models. However, it also raises concerns about the unauthorized exploitation of the dataset, e.g., for commercial propose, which is forbidden by the dataset licenses. In this paper, we introduce a backdoor-based watermarking approach that can be used as a general framework to protect public-available data.
<p align="center">
<img src="https://github.com/Anonymous-Authors-Repo/watermark_dataset/blob/main/pipeline-1.jpg" img width="1000" height="360" />
</p>

## Environment
- `pytorch==1.6.0`
- `torchvision==0.7.0`
- `python==3.6`
- `numpy==1.18.1`
## Pipeline
The watermarking process is as follows. The defender first chooses a target class *C*, and collects a fraction of data from class *C* as the watermarking examples D<sub>wm</sub>. Defenders then apply the adversarial transformation to all samples in D<sub>wm</sub>. Finally, a preset trigger pattern *t* is added to D<sub>wm</sub>. Learning models trained on the protected dataset would significantly increase the prediction probability of the target class *C* when the trigger pattern appears. 
## Image Data
We show the code for Cifar-10 and Caltech256 dataset in Code/Image.
## Text Data
We show the code for SST-2, IMDB and NLI dataset in Code/NLP.
## Audio Data
We show the code for AudioMnist dataset in Code/Audio.

## Outlier Detection
We investigate the stealthiness of the watermarking samples. For image data, we adopt two commonly used autoencoder-based [[code]](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/od_vae_cifar10.html) and confidence-based [[code]](https://github.com/hendrycks/error-detection) outlier detection methods. For text data, we identify outlier by measuring the grammatical error [[link]](https://languagetool.org/) increase rate in watermarking samples. 
