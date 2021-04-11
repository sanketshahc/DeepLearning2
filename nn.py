def tile(input, kernel):
    batch, channel, height, width = input.shape
    kh, kw = kernel  # pooling window, not convolutional kernel
    n_h = height / kh
    n_w = width / kw
    assert height % kh == 0
    assert width % kw == 0

    # after literally 9 hours of fiddling, this works: no idea why...
    # view batch, channel, height, width as:
    #      batch, channel, n height, kernel height, n width, kernel width
    #  permute 0, 1, 2, 4, 3, 5
    tens = input.contiguous()
    tens = tens.view(batch, channel, n_h, kh, n_w, kw)
    tens = tens.permute(0, 1, 2, 4, 3, 5).contiguous()
    tens = tens.view(batch, channel, n_h, n_w, kh * kw)
    return tens, tens.shape[2], tens.shape[3]