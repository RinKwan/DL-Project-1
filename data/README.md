# Data folder

The competition data is **not** included in this repository. Please download it from th website.

```
data/
├── train_images/          50,000 PNG files, 32x32 RGB (train_00000.png ...)
├── train_labels.csv       50,001 lines: id,filename,label
├── test_images/           2,000 PNG files, 32x32 RGB (test_0000.png ...)
├── test.csv               2,001 lines: id,filename
└── sample_submission.csv  2,001 lines: id,label   (the required format)
```

Class indices, as given in `train_labels.csv`:

| label | class | label | class |
|-------|-------|-------|-------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

The first time `dataset.py` reads the images it converts them to one
`uint8` array and saves it in `.cache/` (about 150 MB). Reading 50,000 small
PNG files takes about a minute, and this cache makes every later run start
immediately. Delete the folder if you change the images.

`submission.csv` is written here by `src/predict.py` or
`src/ensemble_predict.py`.
