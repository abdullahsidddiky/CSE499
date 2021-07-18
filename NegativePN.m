a=imread('E166.jpeg')
a=imresize(a,[100 100])
a=rgb2gray(a)
imwrite(a,'e166.bmp')
