# FastMVAE2: On improving and accelerating the fast variational autoencoder-based source separation algorithm for determined mixtures

This repository provides official PyTorch implementation of multichannel variational autoencoder (MVAE) and its fast algorithms proposed in the following papers. The MVAE algorithm in this repo is the same as that previously provided at https://github.com/lili-0805/MVAE.

We also provide pretrained models for speaker-closed and speaker-open situations trained using VCC and WSJ0 datasets, respectively. 

1. Hirokazu Kameoka, Li Li, Shota Inoue, and Shoji Makino, "Supervised Determined Source Separation with Multichannel Variational Autoencoder," Neural Computation, vol. 31, no. 9, pp. 1891-1914, Sep. 2019.
2. Li Li, Hirokazu Kameoka, Shota Inoue, and Shoji Makino, "FastMVAE: A fast optimization algorithm for the multichannel variational autoencoder method," IEEE Accesss, vol. 8, pp. 228740-228753, Dec. 2020.
3. Li Li, Hirokazu Kameoka, and Shoji Makino, "FastMVAE2: On improving and accelerating the fast variational autoencoder-based source separation algorithm for determined mixtures", IEEE TASLP, Oct. 2022.

## Dependencies

Code was tested using following packages. 
A full package list is in requirements.txt.

* Python 3.6.6
* PyTorch 1.10.2
* Scipy 1.5.4
* Numpy 1.19.5

## Download
Get code

```bash
$ git clone https://github.com/lili-0805/mvae-ss.git
```

Using download script to download training dataset, test dataset, and pretrained models.

The test samples were generated using the VCC dataset. Namely, the test samples are speaker-closed for models trained using the VCC dataset, and speaker-open for models trained using the WSJ0 dataset.

Considering the license of WSJ0 database, we do not provide training dataset of WSJ0. Please download WSJ0 database and prepare trainging dataset described in our paper #2 by yourselves. 

```bash
$ cd mvae-ss/exe
$ bash download.sh dataset-VCC
$ bash download.sh test-samples
$ bash download.sh models
```

## Usage

Please use stage to choose model training or test, where stage=0 and stage=1 indicate training and test, respectively.

The following command is the default setting for training ChimeraACVAE source model with the VCC dataset and then testing the trained model with FastMVAE2 algorithm on the downloaded test dataset.

```bash
$ ./run.sh --stage 0 --stop_stage 1 --algorithm FastMVAE2 --dataset vcc --test_mode trained --test_dataset test_input
```

More details are available in the `run.sh` bash file.

## License and citations
License: [MIT](https://choosealicense.com/licenses/mit/)

If you find this work is useful for your research or project, please cite out papers:

```
@article{kameoka2019supervised,
  title={Supervised determined source separation with multichannel variational autoencoder},
  author={Kameoka, Hirokazu and Li, Li and Inoue, Shota and Makino, Shoji},
  journal={Neural computation},
  volume={31},
  number={9},
  pages={1891--1914},
  year={2019},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}
@article{li2020fastmvae,
  title={FastMVAE: A fast optimization algorithm for the multichannel variational autoencoder method},
  author={Li, Li and Kameoka, Hirokazu and Inoue, Shota and Makino, Shoji},
  journal={IEEE Access},
  volume={8},
  pages={228740--228753},
  year={2020},
  publisher={IEEE}
}
@article{li2022fastmvae2,
  title={FastMVAE2: On improving and accelerating the fast variational autoencoder-based source separation algorithm for determined mixtures},
  author={Li, Li and Kameoka, Hirokazu and Makino, Shoji},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2022},
  publisher={IEEE}
}
```

## See also

* Demo: http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/mvae-ss/index.html
* Related work:

1. Underdetermined source separation:
Shogo Seki, Hirokazu Kameoka, Li Li, Tomoki Toda, and Kazuya Takeda, "Underdetermined Source Separation Based on Generalized Multichannel Variational Autoencoder," IEEE Access, vol. 7, No. 1, pp. 168104-168115, Nov. 2019.

2. Determined source separation and dereverberation:
Shota Inoue, Hirokazu Kameoka, Li Li, Shogo Seki, and Shoji Makino, "Joint separation and dereverberation of reverberant mixtures with multichannel variational autoencoder," in Proc. ICASSP2019, pp. 56-60, May 2019.
