# Data preparation

**N.b. The data preparation described in this README have already been performed, you do not need to run it again, and you do not need to download the full dataset. It is provided for reference.**

Kaggle [Letterboxd dataset](https://www.kaggle.com/datasets/gsimonx37/letterboxd)
can be downloaded with (careful: 25GB)

```bash
curl -L -o letterboxd.zip https://www.kaggle.com/api/v1/datasets/download/gsimonx37/letterboxd
```

Unzip:

```bash
unzip letterboxd.zip
```

The notebook [preparation.ipynb](preparation.ipynb) can be used to create
the cleaned up, filtered dataset. All requirements necessary for running this
notebook are in [requirements.txt](requirements.txt).

