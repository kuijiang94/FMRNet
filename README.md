# FMRNet
FMRNet: Image Deraining via Frequency Mutual Revision (AAAI'2024)

[Kui Jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl)

**Paper**: [FMRNet: Image Deraining via Frequency Mutual Revision](https://ojs.aaai.org/index.php/AAAI/article/view/29186)

## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
```


## Evaluation

1. Download the model and place it in `./pretrained_models/`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800) from here and place them in `./Datasets/Synthetic_Rain_Datasets/test/`

3. Run
```
python test.py
```

#### To reproduce PSNR/SSIM scores of the paper, run
```
evaluate_PSNR_SSIM.m 
```

#### Reference:

If you find our code or ideas useful, please cite:

    @article{Jiang2023FMRNet,
        title={Magic ELF: Image Deraining Meets Association Learning and Transformer},
        author={Jiang, Kui and Jiang, Junjun and Liu, Xianming and Xu, Xin and Ma, Xianzheng},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={38},
        number={11},
        pages={12892--12900},
        year={2024}
    }

