# lsd_ext

LSD (Line Segment Detector, von Gioi et al, 2010) is an algorithm for extracting straight lines from gray images. Its original code was published in C language. There was also an OpenCV implementation for it, but that implementation has been removed from OpenCV version 3.4.6 to 3.4.15 and version 4.1.0 to 4.5.3 due original code license conflict. Moreover, after experimenting, I found that there exists some small difference between OpenCV's implementation and the vanilla one, which could cause some drawback in Linelet's metrics on the York Urban dataset.

Therefore, a license-non-conflicting version of LSD for the Python language is required. As far as I know, the community has a repository [pylsd](https://github.com/primetang/pylsd) that does the job. But it doesn't compile original code from scratch, but supply library files directly instead. So I make my own version to suit my needs.

## Install

Windows OS:
```bash
python setup.py build_ext --inplace
```

Linux or Mac OS:
```bash
make
```

## Demo

```bash
python py_lsd.py
```

## Reference

[lsd_1.6.zip](http://www.ipol.im/pub/art/2012/gjmr-lsd/lsd_1.6.zip)
