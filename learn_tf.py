import numpy as np
# import tensorflow as tf
# 
# n = np.array([1,3,4])
# m = np.array([2,4,6,8,10])
# print(m[n])
# phm =  tf.placeholder(tf.float32, [5])
# phn =  tf.placeholder(tf.int32, [3])
# x = tf.nn.embedding_lookup(phm, phn)
# 
# with tf.Session() as sess:
#     x1 = sess.run(x, feed_dict={phm:m,phn:n})
#     print(x1)

#from PIL import Image, ImageDraw, ImageFont
# get an image
#base = Image.open('Pillow/Tests/images/lena.png').convert('RGBA')

# make a blank image for the text, initialized to transparent text color
#txt = Image.new('L', (500,500))
#fnt = ImageFont.truetype('FreeMono.ttf', 20)
#d = ImageDraw.Draw(txt)
#d.text((10,10), "Hello", fill=(255), font=fnt)
#d.text((10,60), "World", fill=(255), font=fnt)

x = np.asarray([[1,2,3],[4,6,5]])
print(np.argmax(x))
print(np.unravel_index(np.argmax(x), x.shape))