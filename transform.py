import numpy as np
from pydicom import read_file
from PIL import Image
from skimage.filters import gaussian
from pydicom.dataset import Dataset, FileMetaDataset
import sys


def normalize_img(img):
    """values in range 0-1"""
    min_v = np.amin(img)
    max_v = np.amax(img)
    if min_v != max_v:
        return (img[:, :] - min_v) / (max_v - min_v)
    return img


def load_normal(file):
    img = np.array(Image.open(file).convert("L"))
    return normalize_img(img), dict()


def load_dicom(file):
    ds = read_file(file)
    return normalize_img(ds.pixel_array), {e.keyword: e.value for e in ds.iterall()}


def load_img(file):
    if file.name.split('.')[-1].lower() == "dcm":
        return load_dicom(file)
    else:
        return load_normal(file)


def resize_to_square(img):
    size = int(np.ceil(np.max(img.shape) * np.sqrt(2)))
    size += 1 - size % 2
    square_img = np.zeros((size, size))
    dy, dx = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    square_img[dy:(img.shape[0] + dy), dx:(img.shape[1] + dx)] = img
    return square_img, (dx, dy, size - img.shape[1] - dx, size - img.shape[0] - dy)


def trim_img(img, dx0, dy0, dx1, dy1):
    trimmed = np.empty((img.shape[0] - dy0 - dy1, img.shape[1] - dx0 - dx1))
    trimmed[:, :] = img[dy0:(dy0 + trimmed.shape[0]), dx0:(dx0 + trimmed.shape[1])]
    return trimmed


def bresenham(x0, y0, x1, y1):
    """generator yielding tuples of xs and ys of pixels traversed on line from (x0, y0) to (x1, y1) (inclusive)"""

    def delta_and_sign(v0, v1):
        return (v1 - v0, 1) if v0 < v1 else (v0 - v1, -1)

    dx, x_sign = delta_and_sign(x0, x1)
    dy, y_sign = delta_and_sign(y0, y1)

    x, y = x0, y0
    yield x, y
    if dx > dy:
        ai = 2 * (dy - dx)
        bi = dy * 2
        d = bi - dx
        while x != x1:
            if d >= 0:
                y += y_sign
                d += ai
            else:
                d += bi
            x += x_sign
            yield x, y
    else:
        ai = 2 * (dx - dy)
        bi = dx * 2
        d = bi - dy
        while y != y1:
            if d >= 0:
                x += x_sign
                d += ai
            else:
                d += bi
            y += y_sign
            yield x, y


def bresenham_list(x0, y0, x1, y1):
    """list of points from bresenham generator"""
    return [*bresenham(x0, y0, x1, y1)]


def points_to_indices(points):
    """points to indices for np.array"""
    return tuple([*zip(*points)][::-1])


def mean_on_line(img, x0, y0, x1, y1):
    """mean of pixels on line from (x0, y0) to (x1, y1) (inclusive)"""
    return np.mean(img[points_to_indices(bresenham_list(x0, y0, x1, y1))])


def calc_radius(img):
    """assume img is already square"""
    r = img.shape[0] // 2
    return r


def rad_to_cart(cx, cy, r, angle):
    """radial to cartesian coordinates conversion, input angle in radians, output x and y truncated ints"""
    x = r * np.cos(angle) + cx
    y = r * np.sin(angle) + cy
    return int(x), int(y)


def lines(r, emitter_angle, detector_count, detector_span):
    """generate lines between emitters and detectors (parallel)"""
    half_span = detector_span / 2.0
    emitter_from = emitter_angle + half_span
    emitter_to = emitter_from - detector_span
    detector_from = emitter_angle + np.pi - half_span
    detector_to = detector_from + detector_span

    emitter_angles = np.linspace(emitter_from, emitter_to, detector_count)
    detector_angles = np.linspace(detector_from, detector_to, detector_count)
    for em_angle, det_angle in zip(emitter_angles, detector_angles):
        ex, ey = rad_to_cart(r, r, r, em_angle)
        dx, dy = rad_to_cart(r, r, r, det_angle)
        yield ex, ey, dx, dy


def emitter_measurements(img, r, emitter_angle, detector_count, detector_span):
    measurements = np.zeros(detector_count)
    for i, line in enumerate(lines(r, emitter_angle, detector_count, detector_span)):
        measurements[i] = mean_on_line(img, *line)
    return measurements


def emitter_angles_range(emitter_step):
    return np.arange(0, np.pi, emitter_step)


def filter_mask(n):
    mask = np.arange(n, dtype=float) - n // 2
    mask[mask % 2 == 0] = 0
    mask[mask % 2 != 0] = (-4 / np.pi ** 2) / (mask[mask % 2 != 0] ** 2)
    mask[n // 2] = 1
    return mask


def convolve(img, n):
    output = np.empty(img.shape)
    for i in range(img.shape[0]):
        output[i, :] = np.convolve(img[i, :], filter_mask(n), mode='same')
    return output


def img_to_sinogram(img, emitter_step, r, detector_count, detector_span):
    angles = emitter_angles_range(emitter_step)
    sinogram = np.empty((angles.shape[0], detector_count))

    for i, emitter_angle in enumerate(angles):
        sinogram[i, :] = emitter_measurements(img, r, emitter_angle, detector_count, detector_span)
    return normalize_img(sinogram)


def emitter_onto_img(img, count, measurements, r, emitter_angle, detector_count, detector_span):
    for i, line in enumerate(lines(r, emitter_angle, detector_count, detector_span)):
        points = points_to_indices(bresenham_list(*line))
        img[points] += measurements[i]
        count[points] += 1


def sinogram_to_img_simple(sinogram, emitter_step, r, detector_count, detector_span, offset):
    """used in experiment"""
    angles = emitter_angles_range(emitter_step)
    img = np.zeros((2 * r + 1, 2 * r + 1))
    count = img.copy()
    for i, emitter_angle in enumerate(angles):
        emitter_onto_img(img, count, sinogram[i, :], r, emitter_angle, detector_count, detector_span)

    used_pixels = count != 0
    img[used_pixels] /= count[used_pixels]
    return normalize_img(trim_img(img, *offset))


def sinogram_to_img_animate(sinogram, emitter_step, r, detector_count, detector_span, offset):
    angles = emitter_angles_range(emitter_step)
    result = []
    img = np.zeros((2 * r + 1, 2 * r + 1))
    count = img.copy()
    for i, emitter_angle in enumerate(angles):
        emitter_onto_img(img, count, sinogram[i, :], r, emitter_angle, detector_count, detector_span)
        frame = img.copy()
        used_pixels = count != 0
        frame[used_pixels] /= count[used_pixels]
        result.append(normalize_img(trim_img(frame, *offset)))
    return np.array(result)


def sinogram_to_img(sinogram, animate, *args):
    """unused, always animate, app decides whether to show it"""
    if animate:
        return sinogram_to_img_animate(sinogram, *args)
    else:
        return sinogram_to_img_simple(sinogram, *args)


def apply_gaussian(images):
    if len(images.shape) == 2:
        return gaussian(images)
    return np.array([gaussian(image) for image in images])


def mean_square_error(input_image, output):
    def mse(output_image):
        return np.square(input_image - output_image).mean()

    if len(output.shape) == 2:
        return mse(output)
    else:
        result = np.empty(output.shape[0])
        for i in range(output.shape[0]):
            result[i] = mse(output[i])
        return result


def root_mean_square_error(input_image, output):
    return mean_square_error(input_image, output) ** 0.5


def empty_dicom():
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 206
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = '1.2.826.0.1.3680043.8.498.51645380419494159785729751472725175471'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.8.498.1'
    file_meta.ImplementationVersionName = 'PYDICOM 2.0.0'

    ds = Dataset()
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.SOPInstanceUID = '1.2.826.0.1.3680043.8.498.51645380419494159785729751472725175471'
    ds.Modality = 'CT'
    ds.StudyInstanceUID = '1.2.826.0.1.3680043.8.498.75112040858074996916346159754932379994'
    ds.SeriesInstanceUID = '1.2.826.0.1.3680043.8.498.64119849432490865623274415908957426618'
    ds.InstanceNumber = "1"
    ds.FrameOfReferenceUID = '1.2.826.0.1.3680043.8.498.10194591012322579188814682575529857631'
    ds.ImagesInAcquisition = "1"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    return ds, file_meta
