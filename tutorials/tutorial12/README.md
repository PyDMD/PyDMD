# CDMD for Background Modeling
__Authors__: [Josh Myers-Dean](https://joshmyersdean.github.io/)

Foreground Modeling using DMD tutorial inspired by [Compressed dynamic mode decomposition for background modeling](https://arxiv.org/abs/1512.04205) by Erichson et al. This tutorial will present the video processing use case treated both with DMD and CDMD.

## Dataset
We will be using the [SegTrackV2 dataset](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html) from Oregon State. This is a binary segmentation dataset with pixel-level annotations and will work well for this tutorial. While some videos only have few frames, there are categories with sufficiently many frames.

## Instructions
1. Open the notebook `dmd.ipynb`
2. Click On `Open in colab`
3. Click `Runtime` -> `Run All` for default configurations
4. mIoU and F1 score plots are saved to the current directory, see cell 15 for filename format. This will allow you to compare different configurations of (c)DMD and their effect on evaluation metrics

## Places to Change
- Try adding noise (cell 9)
- Use regular DMD instead of compressed (cell 10)
- Change the SVD Rank from 0 (optimized fit) to something like 2 or 3 (cell 10)
- Try not using the optimized DMD (cell 10)
- Change compression matrix for compressed DMD (cell 10)
- Change the amount of modes, `K`, to use for static background model (cell 12)
- View how the segmentation visually changes with differing thresholds (cell 16)
- Use a different video! (cell 5, then `Run All`)
