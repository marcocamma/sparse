# sparse
Single file repository with possible implementation of sparse format

This format allows to save in a space efficient manner multi-dimensional images with relatively sparse counts (i.e. many pixels are exactly zero)

Module to convert images to sparse streamable objects
Each image must be a >= 1D array
Each stream is a named tuple with:
   - pixel_idx (index of 1d-flattened image)
   - pixel_value (value of each pixel in 1d-flatted image)
   - frame_idx (of len nimages+1), it indicates the beginning and end of pixel_idx and pixel_value for every image.
   - other "metadata": dtype, nimages, shape, etc. that allows conversion back to standard multidimensional array

Both functional and object oriented interfaces are available

## Functional programming example
```python
    # generate 127 images, each with shape 13,10,17
    imgs1 = generate_random_imgs(shape=(127,13,10,17))
    s1 = images_to_sparse_stream(imgs1)
    # generate more images
    imgs2 = generate_random_imgs(shape=(100,13,10,17))
    s2 = images_to_sparse_stream(imgs2)
    # concatenate_sparse_streams( (s1,s2) ) would also work
    s = concatenate_sparse_streams(s1,s2)
    imgs = sparse_stream_to_images(s)
    save_stream(s,"/tmp/todelete.npy")
    # get one image
    get_image_from_sparse_stream(s,10)
```

## Object-oriented interface
```python
    # generate 127 images, each with shape 13,10,17
    imgs1 = generate_random_imgs(shape=(127,13,10,17))
    s1 = from_images(imgs1)
    s2 = s1[:30] # new SparseStream instance with only the data about the first 30 images
    imgs2 = s2.to_images()
    s2.save("/tmp/todelete")
    s2 = images_to_sparse_stream(imgs2)
    # concatenate, note order matters !
    s = s1 + s2
    # get one image
    s.get_image(10)
    s.to_images(slice_idx=slice(0,20,5))
```
