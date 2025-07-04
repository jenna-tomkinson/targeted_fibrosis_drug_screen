# Illumination correction & whole image quality control (QC)

In this module, we apply a modified version of the CellProfiler illumination correction pipeline from the [cellpainting_predicts_cardiac_fibroblasts](https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis) repository.
We have modified the pipeline to save the illumination correction images (or as CellProfiler names them as function) as `npy` files per channel over correcting the raw TIFF images and save as TIFF images to avoid losing storage space.
For example, each `npy` per channel is a few MB while saving all the corrected TIFF images would be duplicating how many raw images you have (e.g., a 50 GB plate means saving another 50 GB of the corrected plate).

## CellProfiler pipeline

The pipeline that we will be utilizing performs two tasks:

1. Whole image quality control (QC)
2. Correct and save images for illumination errors

### Whole image QC

In the `cellpainting_predicts_cardiac_fibrosis` repository, whole image quality thresholds were already calculated using single-cells from only one plate as an anchor or base to use for all other plates of single-cell data.
Specifically, blur and saturation features extracted for the nuclei and Golgi/plasma membrane channels are used to detect poor quality images as these channels looked to contain the most issues and did best at detecting over-saturated image sets.
In the [evaluation notebook](https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis/blob/main/1.preprocessing_data/1.evaluate_qc.ipynb), a z-scoring method is applied to determine the thresholds in which an image is outside the bounds of being good quality.

These thresholds are added into the illumination correction CellProfiler pipeline under the `FlagImages` module.
If one channel in an image set fails the QC flag, then the whole image set is skipped and not included in the saved corrected images per plate.

For further information on why we originally chose these features and what they measure, please refer to this [README](https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis/blob/main/1.preprocessing_data/README.md) from the `cellpainting_predicts_cardiac_fibrosis` repository.

After the CellProfiler pipeline is ran, the next step is the QC report which will generate a platemap figure to show the distribution of passing FOVs across wells.

### Correct for uneven illumination

Within the CellProfiler pipeline, IC parameters have already been optimized for each channel in the modified Cell Painting.
Images that pass the QC flags will be corrected for uneven illumination and saved in the same bit-depth (16 bit) with a `_illumcorrect` suffix at the end of the image file name.

## Perform IC on data

To calculate, apply, and save images that have been corrected, run the bash script [perform_ic.sh](./perform_ic.sh) using the command below:

```bash
# Make sure your current working dir is the 1.illumination_correction folder
source perform_ic.sh
```

**It took approximately 1 hour and 50 minutes to run illumination correction on 4 plates at the same time with 1,500 image sets (group of channels per FOV) per plate using a Linux-based machine running Pop_OS! LTS 22.04 with an AMD Ryzen 7 3700X 8-Core Processor.**
**There is a total of 16 CPUs with 125 GB of MEM.**
