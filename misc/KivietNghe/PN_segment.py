"""
Translation of PN_segment.m to Python.

Original MATLAB author: Philippe Nghe (16/01/2012)
This version keeps the same step structure and naming as closely as possible.
"""

################################################################################
# %% 

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import (
    exposure,
    filters,
    io,
    measure,
    morphology,
    segmentation,
    util,
)

################################################################################
# %% 

def PN_segphase(imageToSegment, p = None, **kwargs):
    
    #%% ---------------------- PARAMETERS INITIALIZATION ----------------------
    q = {
        # STEP A
        "rangeFiltSize": 35,
        "maskMargin": 5,
        "useFullImage": 0,
        # STEP B
        "LoG_Smoothing": 2,
        "LoG_Threshold": 0,
        "minCellArea": 250,
        # STEP C
        "GaussianFilter": 5,
        "minDepth": 5,
        # STEP E
        "neckDepth": 2,
        # Saving images
        "saveSteps": False,
        "saveDir": os.getcwd(),
    }
    q.update(kwargs)

    if imageToSegment is None:
        raise ValueError("imageToSegment is required")
    
    if p is None:
        p = {}

    #%%  ---------------------- SEGMENTATION ----------------------
    # STEP O (O the letter, not 0 the zero !): suppress shot noise
    O_PhImageFilt = filters.median(imageToSegment, footprint=np.ones((3, 3), dtype=bool))
    # plt.imshow(O_PhImageFilt)

    if _has_field(p, "segmentationFigures"):
        plt.figure(1)
        plt.imshow(O_PhImageFilt, cmap="gray")
        plt.axis("off")
    
    # --------- (MW:) Find the colony outline
    
    # ### (MW:) Find colony region
    
    # STEP A : find a global mask and crop the image
    # (MW:) based on regions of contrast
    A_maskImage = _rangefilt(O_PhImageFilt, int(q["rangeFiltSize"]))
    # plt.imshow(A_maskImage)
    thresh = filters.threshold_otsu(A_maskImage)
    A_maskImage = A_maskImage > thresh
    # plt.imshow(A_maskImage)
    A_maskImage = ndi.binary_fill_holes(A_maskImage)
    A_maskImage = morphology.binary_dilation(A_maskImage, morphology.disk(int(q["maskMargin"])))
    # plt.imshow(A_maskImage)

    labelMaskImage = A_maskImage.astype(bool)
    
    # ### (MW:) Select the right one
    
    # determine region of interest
    labels = measure.label(labelMaskImage)
    propsMaskImage = measure.regionprops(labels)
    # plt.imshow(labels)

    if len(propsMaskImage) == 0:
        # Fallback: no mask was found
        A_cropMaskImage = np.ones_like(imageToSegment, dtype=bool)
        ROI_segmentation = np.array([1, 1, imageToSegment.shape[0], imageToSegment.shape[1]], dtype=int)
    else:
        if _has_field(p, "customColonyCenter"):
            mycenter = np.array(p["customColonyCenter"], dtype=float)
            # MATLAB uses (x, y) for centroids. skimage gives (row, col) = (y, x).
            distances = []
            for r in propsMaskImage:
                y, x = r.centroid
                d = np.sqrt((x - mycenter[0]) ** 2 + (y - mycenter[1]) ** 2)
                distances.append(d)
            idx = int(np.argmin(distances))
        else:
            areas = [r.area for r in propsMaskImage]
            idx = int(np.argmax(areas))

        r = propsMaskImage[idx]
        minr, minc, maxr, maxc = r.bbox

        if q["useFullImage"] != 1:
            A_cropMaskImage = r.image
            # MATLAB ROI is [ymin xmin ymax xmax], 1-based and inclusive.
            ROI_segmentation = np.array([minr + 1, minc + 1, maxr, maxc], dtype=int)
        else:
            A_cropMaskImage = np.ones_like(imageToSegment, dtype=bool)
            ROI_segmentation = np.array([1, 1, imageToSegment.shape[0], imageToSegment.shape[1]], dtype=int)

    y0, x0, y1, x1 = ROI_segmentation
    # Convert MATLAB 1-based inclusive coordinates to Python slices.
    A_cropPhImage = O_PhImageFilt[y0 - 1 : y1, x0 - 1 : x1]
    # plt.imshow(A_cropPhImage)

    if q["saveSteps"]:
        savePNGofImage(A_maskImage, "A_maskImage", q["saveDir"])

    if _has_field(p, "segmentationFigures"):
        plt.figure(1)
        plt.imshow(A_cropPhImage, cmap="gray")
        plt.axis("off")
    
    # --------- (MW:) Segmentation Colony; find edges
    
    # STEP B : find edges
    # (MW:) First polish
    B_negPh = util.invert(_as_float01(A_cropPhImage))
    se = morphology.disk(1)
    B_negPhErode = morphology.erosion(B_negPh, se)
    B_negPh = morphology.reconstruction(B_negPhErode, B_negPh)
    B_negPh = morphology.dilation(B_negPh, se)
    # plt.imshow(B_negPh)

    # MATLAB uses edge(..., 'log', 0, sigma). Canny gives a robust executable analogue.
    
    # (MW:) Ran into issues here, original line was;
    # (MATLAB:)
    # B_edgeImage1 = edge(B_negPh,'log',0,q.Results.LoG_Smoothing); 
    # ie using LoG; but it's not easy to replicate Matlab's behavior
    
    # Use LoG zero-crossings with threshold gating, similar to MATLAB edge(..., 'log', thresh, sigma).
    edges_log = sk.filters.gaussian(B_negPh, sigma=3)
    edges_log = sk.filters.laplace(edges_log)
    edges_min = ndimage.minimum_filter(edges_log, footprint=sk.morphology.disk(1))
    edges_pos = ndimage.maximum_filter(edges_log, footprint=sk.morphology.disk(1))
    B_edgeImage1 = np.logical_and(edges_min < 0, edges_pos >0)
        # plt.imshow(B_edgeImage1)

    # suppress noisy surroundings
    if A_cropMaskImage.shape != B_edgeImage1.shape:
        # Should not happen in normal flow, but keep executable if shapes diverge.
        A_cropMaskImage = np.ones_like(B_edgeImage1, dtype=bool)
    B_edgeImage2 = B_edgeImage1 & A_cropMaskImage
    B_fillEdgeImage2 = ndi.binary_fill_holes(B_edgeImage2)
    B_fillEdgeImage2 = morphology.remove_small_objects(
        B_fillEdgeImage2.astype(bool), min_size=int(q["minCellArea"]), connectivity=1
    )
    B_edgeImage2 = B_edgeImage1 & B_fillEdgeImage2
    B_edgeImage2 = morphology.remove_small_objects(B_edgeImage2.astype(bool), min_size=30)
        # plt.imshow(B_edgeImage2)

    DE_boolean = 0
    if DE_boolean:
        Btmp = B_negPh.copy()
        Btmp[Btmp < np.mean(Btmp)] = 0
        mask_fromintensity = Btmp.astype(bool)
        se5 = morphology.disk(10)
        mask_fromintensity2 = util.invert(morphology.erosion(util.invert(mask_fromintensity), se5))

        B_edgeImage3 = B_edgeImage2 & mask_fromintensity2
        B_fillEdgeImage3 = ndi.binary_fill_holes(B_edgeImage3)
        B_fillEdgeImage3 = morphology.remove_small_objects(
            B_fillEdgeImage3.astype(bool), min_size=int(q["minCellArea"]), connectivity=1
        )
        B_edgeImage3 = B_edgeImage3 & B_fillEdgeImage3
        B_edgeImage3 = morphology.remove_small_objects(B_edgeImage3.astype(bool), min_size=30)
        B_edgeImage2 = B_edgeImage3

    if q["saveSteps"]:
        savePNGofImage(B_edgeImage1, "B_edgeImage1", q["saveDir"])
        savePNGofImage(B_edgeImage2, "B_edgeImage2", q["saveDir"])
        savePNGofImage(B_fillEdgeImage2, "B_fillEdgeImage2", q["saveDir"])

    if _has_field(p, "segmentationFigures"):
        plt.figure(1)
        plt.imshow(B_edgeImage1, cmap="gray")
        plt.axis("off")
        plt.figure(2)
        plt.imshow(B_edgeImage2, cmap="gray")
        plt.axis("off")

    # --------- (MW:) Segment; watershedding

    # STEP C : prepare seeds for watershedding
    C_smoothPh = ndi.gaussian_filter(_as_float01(A_cropPhImage), sigma=float(q["GaussianFilter"]))
    C_localMinPh = morphology.h_minima(C_smoothPh, h=float(q["minDepth"])) & B_fillEdgeImage2
        # plt.imshow(A_cropPhImage)
        # plt.imshow(C_localMinPh)
        # plt.imshow(B_fillEdgeImage2)

    # MATLAB: imfill(B_edgeImage2, find(C_localMinPh))
    C_cellMask = _fill_holes_with_seed_points(B_edgeImage2, C_localMinPh)
    C_cellMask = morphology.binary_opening(C_cellMask)
        # plt.imshow(C_cellMask)

    # shrinking steps to cut some cells
    C_seeds1 = C_cellMask & (~B_edgeImage2)
    C_seeds2 = morphology.binary_opening(C_seeds1)
    C_seeds2 = morphology.thin(C_seeds2)

    if q["saveSteps"]:
        savePNGofImage(C_cellMask & (~C_localMinPh), "C_Mask and minima", q["saveDir"])
        savePNGofImage(C_seeds1, "C_seeds1", q["saveDir"])
        savePNGofImage(C_seeds2, "C_seeds2", q["saveDir"])

    if _has_field(p, "segmentationFigures"):
        plt.figure(1)
        plt.imshow(C_cellMask, cmap="gray")
        plt.axis("off")
        plt.figure(2)
        plt.imshow(C_seeds2, cmap="gray")
        plt.axis("off")

    # STEP D : suppress branch points of the skeleton
    brchpts = PN_FindBranchPoints(C_seeds2)
    C_seeds2 = C_seeds2.copy()
    C_seeds2[brchpts.astype(bool)] = 0

    # some cleaning
    C_seeds2 = _remove_spurs(C_seeds2, iterations=3)
    C_seeds2 = morphology.remove_small_objects(C_seeds2.astype(bool), min_size=10, connectivity=2)

    if _has_field(p, "segmentationFigures"):
        plt.figure(1)
        plt.imshow(C_seeds2, cmap="gray")
        plt.axis("off")

    # STEP E : cut long cells which neck is deeper than neckDepth
    continueToCut = True

    icut = C_seeds2.copy()
    while continueToCut:
        cellsToRemove, cutPoints = PN_CutLongCells(icut, C_cellMask, q["neckDepth"])
        if np.max(cutPoints) == 0:
            continueToCut = False
        else:
            cutPoints = morphology.binary_dilation(cutPoints.astype(bool), morphology.disk(2))
            C_seeds2[cutPoints] = False
            icut[cutPoints] = False
            icut = icut & (~cellsToRemove)

    if q["saveSteps"]:
        savePNGofImage(C_seeds2, "C_seeds2", q["saveDir"])

    if _has_field(p, "segmentationFigures"):
        plt.figure(1)
        plt.imshow(C_seeds2, cmap="gray")
        plt.axis("off")

    # STEP Z : final segmentation by watershedding
    Z_maskToFill = morphology.binary_dilation(C_cellMask)
    Z_background = ~Z_maskToFill

    Z_seeds = _remove_spurs(C_seeds2, iterations=3)
    Z_seeds = morphology.remove_small_objects(Z_seeds.astype(bool), min_size=4, connectivity=2)

    Z_d1 = -ndi.distance_transform_edt(Z_background)

    # MATLAB imimposemin equivalent by marker-based watershed.
    seed_markers = measure.label(Z_seeds)
    background_marker = seed_markers.max() + 1
    markers = seed_markers.copy()
    markers[Z_background] = background_marker

    Z_segmentedImage = segmentation.watershed(Z_d1, markers=markers)
    Z_segmentedImage[Z_background] = 0

    # MATLAB bwareaopen on label image behaves as cleaning nonzero foreground.
    Z_fg = morphology.remove_small_objects((Z_segmentedImage > 0), min_size=10)
    Z_segmentedImage = measure.label(Z_fg)

    Z_segmentedImage = ndi.grey_dilation(Z_segmentedImage, footprint=morphology.diamond(1))

    # re-suppress small cells
    Z_segmentedImage = MW_removesmallsegmented(Z_segmentedImage, int(q["minCellArea"]))

    if q["saveSteps"]:
        savePNGofImage(Z_maskToFill & (~Z_seeds), "Z_Mask and seeds", q["saveDir"])
        savePNGofImage(Z_segmentedImage, "Z_segmentedImage", q["saveDir"])

    if _has_field(p, "segmentationFigures"):
        plt.figure(1)
        plt.imshow(Z_segmentedImage, cmap="nipy_spectral")
        plt.axis("off")

    if np.max(Z_segmentedImage) == 0:
        print(" * !! WATCH OUT !! no cells found on this frame...")

    return A_cropPhImage, Z_segmentedImage, ROI_segmentation

################################################################################
# %% 

def savePNGofImage(image, name, saveDirectory):
    filename = os.path.join(saveDirectory, f"{name}.png")
    if image is None or np.size(image) == 0:
        image = np.array([0], dtype=np.uint8)

    arr = np.asarray(image)
    if arr.dtype == bool:
        out = (arr.astype(np.uint8) * 255)
    elif np.issubdtype(arr.dtype, np.floating):
        out = exposure.rescale_intensity(arr, out_range=(0, 255)).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.integer):
        if arr.max() <= np.iinfo(np.uint8).max:
            out = arr.astype(np.uint8)
        elif arr.max() <= np.iinfo(np.uint16).max:
            out = arr.astype(np.uint16)
        else:
            out = exposure.rescale_intensity(arr.astype(np.float32), out_range=(0, 65535)).astype(np.uint16)
    else:
        out = exposure.rescale_intensity(arr.astype(np.float32), out_range=(0, 255)).astype(np.uint8)

    io.imsave(filename, out)


def PN_FindBranchPoints(skeleton):
    skel = skeleton.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    n_neighbors = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0) - skel.astype(np.uint8)
    return skel & (n_neighbors > 2)


def PN_CutLongCells(icut, C_cellMask, neckDepth):
    # Placeholder equivalent for external MATLAB helper not present in repository.
    # Returns no cut points, preserving executable behavior.
    cellsToRemove = np.zeros_like(icut, dtype=bool)
    cutPoints = np.zeros_like(icut, dtype=bool)
    _ = C_cellMask
    _ = neckDepth
    return cellsToRemove, cutPoints


def MW_removesmallsegmented(label_img, min_area):
    labels = np.asarray(label_img)
    if labels.size == 0:
        return labels

    out = labels.copy()
    props = measure.regionprops(out)
    for r in props:
        if r.area < min_area:
            out[out == r.label] = 0

    out = measure.label(out > 0)
    return out


def _remove_spurs(binary_img, iterations=1):
    # Approximation of MATLAB bwmorph(..., 'spur', n):
    # iteratively remove endpoints from skeleton-like structures.
    out = binary_img.astype(bool).copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(int(iterations)):
        neighbors = ndi.convolve(out.astype(np.uint8), kernel, mode="constant", cval=0) - out.astype(np.uint8)
        endpoints = out & (neighbors <= 1)
        out[endpoints] = False
    return out


def _fill_holes_with_seed_points(edge_img, seed_points):
    edge_img = edge_img.astype(bool)
    seed_points = seed_points.astype(bool)

    filled = ndi.binary_fill_holes(edge_img)
    holes = filled & (~edge_img)
    hole_labels = measure.label(holes)

    labels_at_seed = np.unique(hole_labels[seed_points])
    labels_at_seed = labels_at_seed[labels_at_seed > 0]

    selected_holes = np.isin(hole_labels, labels_at_seed)
    return edge_img | selected_holes


def _rangefilt(image, size):
    # MATLAB rangefilt analogue: local max - local min.
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1

    img = _as_float01(image)
    maxf = ndi.maximum_filter(img, size=size)
    minf = ndi.minimum_filter(img, size=size)
    return maxf - minf


def _edge_log_like_matlab(image, sigma=2.0, thresh=0, size=3):
    # LoG response followed by zero-crossing detection and contrast thresholding.
    img = _as_float01(image).astype(np.float32, copy=False)
    log_response = ndi.gaussian_laplace(img, sigma=float(sigma))

    # Running in fine-tune issues, mostly negative values    
    # maxmin=np.percentile(np.abs(log_response), 90)
    # plt.imshow(log_response, cmap="seismic", vmin=(-1*maxmin), vmax=maxmin)
    
    # plt.hist(np.ravel(log_response))
    
    zero_crossings, local_contrast = _log_zero_crossings_and_contrast(log_response, size=size)
        # plt.imshow(zero_crossings)
    edge_threshold = _resolve_log_threshold(local_contrast, zero_crossings, thresh)
    return zero_crossings & (local_contrast >= edge_threshold)


def _log_zero_crossings_and_contrast(log_response, size=3):
    # A zero crossing exists if 3x3 neighborhood spans both negative and positive values.
    max_nb = ndi.maximum_filter(log_response, size=size, mode="nearest")
    min_nb = ndi.minimum_filter(log_response, size=size, mode="nearest")
    zero_crossings = (max_nb > 0) & (min_nb < 0)
    local_contrast = max_nb - min_nb

    # Suppress border artifacts caused by incomplete neighborhoods.
    zero_crossings[[0, -1], :] = False
    zero_crossings[:, [0, -1]] = False
    return zero_crossings, local_contrast


def _resolve_log_threshold(local_contrast, zero_crossings, thresh):
    if thresh is None or float(thresh) == 0.0:
        vals = local_contrast[zero_crossings]
        if vals.size == 0:
            return np.inf
        # Heuristic auto-threshold when user requests MATLAB-like automatic mode.
        return float(np.percentile(vals, 80))
    return float(thresh)


def _as_float01(image):
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    return util.img_as_float(arr)


def _has_field(p, key):
    if p is None:
        return False
    if isinstance(p, dict):
        return key in p
    return hasattr(p, key)

# %%
