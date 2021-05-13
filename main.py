import transform
import streamlit as st
import numpy as np
import os
import tempfile
from base64 import b64encode
from PIL import Image
from pydicom.dataset import FileDataset


MASK_SIZE = 11
SINOGRAM_GIF_DURATION = 15
OUTPUT_GIF_DURATION = 30


class MetaData:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.value = None
        self.input = None


META_DATA = [
    MetaData('PatientName', 'Patient\'s name'),
    MetaData('PatientID', 'Patient\'s ID'),
    MetaData('PatientSex', 'Patient\'s sex'),
    MetaData('PatientBirthDate', 'Patient\'s birth date'),
    MetaData('StudyDate', 'Study date'),
    MetaData('ImageComments', 'Comment')
]


REQUIRED_META_DATA = set(data.name for data in META_DATA)


class Result:
    def __init__(self, input_image, meta_data, sinogram, filtered_sinogram, output_images, rmse):
        self.input_image = input_image
        self.meta_data = meta_data
        self.sinogram = sinogram
        self.filtered_sinogram = filtered_sinogram
        self.output_images = output_images
        self.rmse = rmse


def file_to_base64(filename, content):
    with open(filename, "rb") as file:
        data = file.read()
        encoded = b64encode(data).decode()
    os.remove(filename)
    return f"data:{content};base64,{encoded}"


@st.cache(show_spinner=True)
def apply_transformation(file, emitter_step, detector_count, detector_span, apply_filter, apply_gaussian):
    input_image, meta_data = transform.load_img(file)
    meta_data = {name: meta_data.get(name, "-") for name in REQUIRED_META_DATA}
    img, offset = transform.resize_to_square(input_image)
    r = transform.calc_radius(img)
    emitter_step_rad = np.deg2rad(emitter_step)
    detector_span_rad = np.deg2rad(detector_span)
    sinogram = transform.img_to_sinogram(img, emitter_step_rad, r, detector_count,
                                         detector_span_rad)
    if apply_filter:
        filtered_sinogram = transform.convolve(sinogram, MASK_SIZE)
        used_sinogram = filtered_sinogram
    else:
        filtered_sinogram = None
        used_sinogram = sinogram
    output_images = transform.sinogram_to_img_animate(used_sinogram, emitter_step_rad,
                                                      r, detector_count, detector_span_rad,
                                                      offset)
    if apply_gaussian:
        output_images = transform.apply_gaussian(output_images)
    rmse = transform.root_mean_square_error(input_image, output_images)
    return Result(input_image, meta_data, sinogram,
                  transform.normalize_img(filtered_sinogram),
                  output_images, rmse)


def show_input_img(result):
    expander = st.beta_expander("Input")
    col1, col2 = expander.beta_columns(2)
    col1.image(result.input_image, "Input image", width=300)
    for data in META_DATA:
        col2.markdown(f"**{data.label}**: {result.meta_data[data.name]}")


@st.cache
def animated_sinogram(sinogram):
    def sinogram_image(rows):
        sinogram_copy = np.zeros(sinogram.shape)
        sinogram_copy[:rows, :] = sinogram[:rows, :]
        return Image.fromarray(np.uint8(sinogram_copy * 255))

    frames = [sinogram_image(0)]
    for i in range(min(200, sinogram.shape[0])):
        frames.append(sinogram_image(i + 1))
    filename = tempfile.NamedTemporaryFile(suffix=".gif").name
    frames[0].save(filename, format="GIF", append_images=frames[1:],
                   save_all=True, duration=SINOGRAM_GIF_DURATION, loop=0)
    return file_to_base64(filename, "file/gif")


def show_sinogram(result, animate):
    expander = st.beta_expander("Sinogram")
    col1, col2 = expander.beta_columns(2)
    img_elem = col1.empty()
    if animate:
        img_elem.markdown(f"<img src='{animated_sinogram(result.sinogram)}' width=300/>",
                          unsafe_allow_html=True)
    else:
        img_elem.image(result.sinogram, width=300)

    if result.filtered_sinogram is not None:
        col2.image(result.filtered_sinogram, "Filtered sinogram", width=300)


def save_dcm(image):
    filename = tempfile.NamedTemporaryFile(suffix=".dcm").name
    ds, meta = transform.empty_dicom()
    ds.Rows = image.shape[0]
    ds.Columns = image.shape[1]
    ds.PixelData = np.asarray(image * 255, dtype=np.uint8).tobytes()
    ds.PatientName = META_DATA[0].input
    ds.PatientID = META_DATA[1].input
    ds.PatientSex = META_DATA[2].input
    ds.PatientBirthDate = META_DATA[3].input
    ds.StudyDate = META_DATA[4].input
    ds.ImageComments = META_DATA[5].input
    fds = FileDataset(filename, ds, file_meta=meta, preamble=b"\0" * 128)
    fds.is_implicit_VR = False
    fds.is_little_endian = True
    fds.save_as(filename)

    href = file_to_base64(filename, "file/dcm")
    return f'<a href="{href}" download="output.dcm">Download File</a>'


@st.cache
def animated_output_image(images):
    frames = []
    di = 1 if images.shape[0] < 200 else int(images.shape[0] / 200)
    for image in images[np.arange(0, images.shape[0], di)]:
        frames.append(Image.fromarray(np.uint8(image * 255)))
    frames.append(Image.fromarray(np.uint8(images[-1] * 255)))
    filename = tempfile.NamedTemporaryFile(suffix=".gif").name
    frames[0].save(filename, format="GIF", append_images=frames[1:],
                   save_all=True, duration=OUTPUT_GIF_DURATION, loop=0)
    return file_to_base64(filename, "file/gif")


def show_output_image(result, animate):
    expander = st.beta_expander("Output")
    col1, col2 = expander.beta_columns(2)

    img_elem = col1.empty()
    if animate:
        img_elem.markdown(f"<img src='{animated_output_image(result.output_images)}' width=300/>",
                          unsafe_allow_html=True)
    else:
        img_elem.image(result.output_images[-1], width=300)

    col1.markdown("**Root mean squared error**")
    col1.line_chart(result.rmse)

    for data in META_DATA:
        data.input = col2.text_input(data.label, result.meta_data[data.name],
                                     key=f"text-input-{data.name}")

    if expander.button("Generate DICOM file"):
        link = save_dcm(result.output_images[-1])
        expander.markdown(link, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Computed Tomography Simulator")
    st.title("Computed Tomography Simulator")
    params_expander = st.sidebar.beta_expander("Parameters", expanded=True)
    emitter_step = params_expander.number_input("Emitter step (deg.)", min_value=0.25,
                                                max_value=179.75, step=0.25, value=1.0)
    detector_count = params_expander.number_input("Detector count", min_value=1, max_value=720,
                                                  step=1, value=90)
    detector_span = params_expander.number_input("Detector span (deg.)", min_value=0.25,
                                                 max_value=270.0, step=0.25, value=180.0)
    apply_filter = params_expander.checkbox("Convolution filter")
    apply_gaussian = params_expander.checkbox("Gaussian filter")
    animate = params_expander.checkbox("Animate")

    file = st.sidebar.file_uploader("Choose file", type=('png', 'jpg', 'dcm'))
    if file is not None:
        result = apply_transformation(file, emitter_step, detector_count,
                                      detector_span, apply_filter, apply_gaussian)
        show_input_img(result)
        show_sinogram(result, animate)
        show_output_image(result, animate)
    else:
        st.text("No file")


if __name__ == '__main__':
    main()
