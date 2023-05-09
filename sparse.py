""" 
Module to convert images to sparse streamable objects
Each image must be a >= 1D array
Each stream is a named tuple with:
   - pixel_idx (index of 1d-flattened image)
   - pixel_value (value of each pixel in 1d-flatted image)
   - frame_idx (of len nimages+1), it indicates the beginning and end of pixel_idx and pixel_value for every image.
   - other "metadata": dtype, nimages, shape, etc. that allows conversion back to standard multidimensional array

Both functional and object oriented interfaces are available

Functional programming example
    # generate 127 images, each with shape 13,10,17
    imgs1 = generate_random_imgs(shape=(127,13,10,17))
    s1 = images_to_sparse_stream(imgs1
    # generate more images
    imgs2 = generate_random_imgs(shape=(100,13,10,17))
    s2 = images_to_sparse_stream(imgs2)
    # concatenate_sparse_streams( (s1,s2) ) would also work
    s = concatenate_sparse_streams(s1,s2)
    imgs = sparse_stream_to_images(s)
    save_stream(s,"/tmp/todelete.npy")
    # get one image
    get_image_from_sparse_stream(s,10)

Object-oriented interface
    # generate 127 images, each with shape 13,10,17
    imgs1 = generate_random_imgs(shape=(127,13,10,17))
    s1 = from_images(imgs)
    s2 = s1[:30] # new SparseStream instance with only the first 30 images
    imgs2 = s2.to_images()
    s2.save("/tmp/todelete")
    s2 = images_to_sparse_stream(imgs2)
    # concatenate, note order matters !
    s = s1 + s2
    # get one image
    s.get_image(10)
    s.to_images(slice_idx=slice(0,20,5))

"""
import numpy as np
from collections import namedtuple

_sparse_stream = namedtuple(
    "sparse_stream", ["pixel_value", "pixel_idx", "frame_idx", "shape", "dtype"]
)


__all__ = [
    "images_to_sparse_stream",
    "concatenate_sparse_streams",
    "save_stream",
    "load_stream",
    "sparse_stream_to_images",
    "SparseStream",
    "slice_stream",
]


class sparse_stream(_sparse_stream):
    @property
    def nimages(self):
        return len(self.frame_idx) - 1

    @property
    def size_one_image(self):
        s = 1
        for npix in self.shape:
            s = s * npix
        return s

    @property
    def size(self):
        return self.nimages * self.size_one_image


def ravel(imgs):
    return imgs.reshape((imgs.shape[0], -1))


def unravel(imgs, shape):
    return imgs.reshape((-1,) + tuple(shape))


def compare_streams(s1, s2):
    if id(s1) == id(s2):
        return True
    else:
        return all([np.alltrue(getattr(s1, k) == getattr(s2, k)) for k in s1._fields])


def images_to_sparse_stream(imgs, t0=0):
    """return sparse_stream (frame_idx will be nimages+1 long)"""
    shape_one_image = imgs.shape[1:]
    imgs = ravel(imgs)
    t, pixel_idx = np.nonzero(imgs > 0)
    counts = imgs[t, pixel_idx]
    if t0 != 0:
        t += t0
    frame_idx = np.bincount(t, minlength=imgs.shape[0])
    frame_idx = np.cumsum(frame_idx)
    frame_idx = np.concatenate(((t0,), frame_idx))
    return sparse_stream(
        pixel_value=counts,
        pixel_idx=pixel_idx,
        frame_idx=frame_idx,
        shape=shape_one_image,
        dtype=imgs.dtype,
    )


def concatenate_sparse_streams(*streams):
    if len(streams) == 1 and not isinstance(streams[0], sparse_stream):
        streams = streams[0]
    if len(streams) == 0:
        raise ValueError("No sparse_streams provided")
    elif len(streams) == 1 and isinstance(streams[0], sparse_stream):
        return streams[0]
    else:
        ret = dict()
        assert all([s.shape == streams[0].shape for s in streams])
        assert all([s.dtype == streams[0].dtype for s in streams])
        for key in "pixel_value", "pixel_idx":
            ret[key] = np.concatenate([getattr(s, key) for s in streams])
        ret["shape"] = streams[0].shape
        ret["dtype"] = streams[0].dtype
        frame_idx = streams[0].frame_idx
        for s in streams[1:]:
            n = frame_idx[-1]
            # print(s.frame_idx[1:] + n)
            frame_idx = np.concatenate((frame_idx, s.frame_idx[1:] + n))
        ret["frame_idx"] = frame_idx

        ret = sparse_stream(**ret)
    return ret


def save_stream(stream, filename):
    tosave = stream._asdict()
    np.save(filename, tosave)


def load_stream(filename):
    data = np.load(filename, allow_pickle=True).item()
    return sparse_stream(*data.values())


def sparse_stream_to_images(stream, as_counts=True, slice_idx=None):
    """
    if as_counts, each non zero pixel has its corresponding count (else it is just 1)
    """
    dtype = stream.pixel_value.dtype if as_counts else bool
    range_imgs = range(stream.nimages)
    if slice_idx is not None:
        range_imgs = range_imgs[slice_idx]
    imgs = np.zeros(len(range_imgs) * stream.size_one_image, dtype=dtype)
    for i, n in enumerate(range_imgs):
        offset = stream.size_one_image * i
        idx = slice(*stream.frame_idx[n : n + 2])
        if as_counts:
            imgs[offset + stream.pixel_idx[idx]] = stream.pixel_value[idx]
        else:
            imgs[offset + stream.pixel_idx[idx]] = True
    return unravel(imgs, stream.shape)


def get_image_from_sparse_stream(stream, frame, as_counts=True):
    """
    if as_counts, each non zero pixel has its corresponding count (else it is just 1)
    """
    idx = slice(frame, frame + 1)
    img = sparse_stream_to_images(stream, as_counts=as_counts, slice_idx=idx)[0]
    return img


def slice_stream(stream, frames):
    if isinstance(frames, int):
        frames = slice(frames, frames + 1)
    range_imgs = range(stream.nimages)[frames]
    assert (
        range_imgs.step == 1
    ), "stream slicing only implemented for continous slices to avoid array copying"
    frame_start = range_imgs.start
    frame_stop = range_imgs.stop
    if frame_stop == 0:
        frame_stop = frame_start + 1  # in case frames == -1
    idx = slice(stream.frame_idx[frame_start], stream.frame_idx[frame_stop])
    # print(frame_start,frame_stop,idx)
    return sparse_stream(
        pixel_value=stream.pixel_value[idx],
        pixel_idx=stream.pixel_idx[idx],
        frame_idx=stream.frame_idx[frame_start : frame_stop + 1]
        - stream.frame_idx[frame_start],
        shape=stream.shape,
        dtype=stream.dtype,
    )


class SparseStream:
    def __init__(self, sparse_stream_instance=None, **kw):
        if sparse_stream_instance is None and len(kw) != 0:
            sparse_stream_instance = sparse_stream(**kw)
        self.stream = sparse_stream_instance

    def from_images(self, imgs):
        self.stream = images_to_sparse_stream(imgs)
        return self.stream

    def to_images(self, as_counts=True, slice_idx=None):
        return sparse_stream_to_images(
            self.stream, as_counts=as_counts, slice_idx=slice_idx
        )

    def get_image(self, frame):
        return get_image_from_sparse_stream(self.stream, frame)

    def save(self, filename):
        return save_stream(self.stream, filename)

    def load(self, filename):
        self.stream = load_stream(filename)

    def __getitem__(self, idx):
        return SparseStream(slice_stream(self.stream, idx))

    def __add__(self, other):
        s = concatenate_sparse_streams(self.stream, other.stream)
        return SparseStream(s)

    def __repr__(self):
        if self.stream is None:
            return f"SparseStream, Empty"
        else:
            s = self.stream
            return f"SparseStream, {s.nimages} × {s.shape}, type: {s.dtype}"


def from_images(imgs):
    sparse_stream = images_to_sparse_stream(imgs)
    return SparseStream(sparse_stream)


def generate_random_imgs(average_count=0.3, shape=(200, 128, 256)):
    imgs = np.random.poisson(average_count, size=shape)
    return imgs


def test1(shape=(200, 128, 256)):
    """test imgs->stream->imgs"""
    imgs = generate_random_imgs(shape=shape)
    stream = images_to_sparse_stream(imgs)
    imgs2 = sparse_stream_to_images(stream)
    assert np.alltrue(imgs == imgs2)


def test2(shape=(200, 128, 256)):
    """test concatenate_sparse_streams arguments"""
    imgs = generate_random_imgs(shape=shape)
    n = shape[0] // 3
    stream1 = images_to_sparse_stream(imgs[:n])
    stream2 = images_to_sparse_stream(imgs[n:], t0=n)
    stream_c1 = concatenate_sparse_streams(stream1, stream2)
    stream_c2 = concatenate_sparse_streams((stream1, stream2))
    assert compare_streams(stream_c1, stream_c2)


def test3(shape=(200, 128, 256)):
    """test concatenate_sparse_streams"""
    imgs = generate_random_imgs(shape=shape)
    n = shape[0] // 3
    stream1 = images_to_sparse_stream(imgs[:n])
    stream2 = images_to_sparse_stream(imgs[n:])
    stream = concatenate_sparse_streams(stream1, stream2)
    imgs2 = sparse_stream_to_images(stream)
    assert np.alltrue(imgs == imgs2)


def test4(shape=(200, 128, 256)):
    """test concatenate_sparse_streams for 3 streams"""
    imgs = generate_random_imgs(shape=shape)
    n1 = shape[0] // 3
    n2 = n1 * 2
    stream1 = images_to_sparse_stream(imgs[:n1])
    stream2 = images_to_sparse_stream(imgs[n1:n2])
    stream3 = images_to_sparse_stream(imgs[n2:])
    stream = concatenate_sparse_streams(stream1, stream2, stream3)
    imgs2 = sparse_stream_to_images(stream)
    assert np.alltrue(imgs == imgs2)


def test5(shape=(200, 128, 256)):
    """test save/reload"""
    imgs = generate_random_imgs(shape=shape)
    stream = images_to_sparse_stream(imgs)
    save_stream(stream, "/tmp/sparse_saving_text.npy")
    stream2 = load_stream("/tmp/sparse_saving_text.npy")
    assert compare_streams(stream, stream2)


def test6(shape=(200, 10, 12, 256)):
    """test 3D array per image, it could be useful for multi module detectors"""
    imgs = generate_random_imgs(shape=shape)
    n1 = shape[0] // 3
    n2 = n1 * 2
    stream1 = images_to_sparse_stream(imgs[:n1])
    stream2 = images_to_sparse_stream(imgs[n1:n2])
    stream3 = images_to_sparse_stream(imgs[n2:])
    stream = concatenate_sparse_streams(stream1, stream2, stream3)
    imgs2 = sparse_stream_to_images(stream)
    assert np.alltrue(imgs == imgs2)


def test7(shape=(200, 10 * 12 * 256)):
    """test 1D array per image"""
    imgs = generate_random_imgs(shape=shape)
    n1 = shape[0] // 3
    n2 = n1 * 2
    stream1 = images_to_sparse_stream(imgs[:n1])
    stream2 = images_to_sparse_stream(imgs[n1:n2])
    stream3 = images_to_sparse_stream(imgs[n2:])
    stream = concatenate_sparse_streams(stream1, stream2, stream3)
    imgs2 = sparse_stream_to_images(stream)
    assert np.alltrue(imgs == imgs2)


def test8(shape=(200, 10 * 12 * 256)):
    """test image slicing"""
    imgs = generate_random_imgs(shape=shape)
    imgs_idx = slice(10, len(imgs) - 10, 7)
    stream = images_to_sparse_stream(imgs)
    imgs2 = sparse_stream_to_images(stream, slice_idx=imgs_idx)
    assert np.alltrue(imgs[imgs_idx] == imgs2)


def test9(shape=(200, 10, 12, 256)):
    """test object oriented interface (slicing)"""
    s1 = SparseStream()
    imgs = generate_random_imgs(shape=shape)
    s1.from_images(imgs)
    s2 = s1[:20]
    imgs2 = s2.to_images()
    assert np.alltrue(imgs[:20] == imgs2)


def test10(shape=(200, 10, 12, 256)):
    """test object oriented interface concatenation"""
    n1 = shape[0] // 3
    imgs = generate_random_imgs(shape=shape)
    s1 = from_images(imgs[:n1])
    s2 = from_images(imgs[n1:])
    s = s1 + s2
    imgs2 = s.to_images()
    assert np.alltrue(imgs == imgs2)


def test11(shape=(200, 10, 12, 256)):
    """test object oriented interface"""
    n1 = shape[0] // 3
    imgs = generate_random_imgs(shape=shape)
    s1 = from_images(imgs[:n1])
    s2 = from_images(imgs[n1:])
    s = s1 + s2
    imgs2 = s.to_images()
    assert np.alltrue(imgs == imgs2)


def test_all():
    keys = globals()
    for k in keys:
        f = globals()[k]
        if callable(f) and k.find("test") == 0 and k != "test_all":
            print("testing", k)
            f()


if __name__ == "__main__":
    pass
    test_all()
