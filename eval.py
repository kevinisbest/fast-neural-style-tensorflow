# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
from tkinter import *
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk
from PIL import Image
import numpy as np
import os

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
# tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
# tf.app.flags.DEFINE_string("image_file", "a.jpg", "")

FLAGS = tf.app.flags.FLAGS


def main(_):

    global image_compress
    global srcw
    global srch

    global panelA 

    global panelB
    
    # Get image's height and width.
    height = 0
    width = 0
    # image_compress.show()
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            # Make sure 'generated' directory exists.
            generated_file = 'generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_file)

    image_compress = ImageTk.PhotoImage(image_compress)
    
    show()


def show():
    global panelA 
    global panelB
    
    if(os.path.exists('generated/res.jpg')):
        edged = Image.open("generated/res.jpg")
            

    edged = edged.resize( (srcw, srch), Image.BILINEAR )
    edged = ImageTk.PhotoImage(edged)
    if panelA is None or panelB is None:
        # the first panel will store our original image
        panelA = Label(image=image_compress,text = 'source')
        panelA.image = image_compress
        print('in if')
        panelA.pack(side="left", padx=10, pady=10)
        text = Label(window,text="source  ").place(x=15, y=83)
        
        # while the second panel will store the edge map
        panelB = Label(image=edged,text = 'out')
        panelB.image = edged
        panelB.pack(side="right", padx=10, pady=10)
        
    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image_compress)
        panelB.configure(image=edged)
        panelA.image = image_compress
        panelB.image = edged


def select_image():
    # grab a reference to the image panels
    
    global image_compress
    global srcw
    global srch

    
    path = filedialog.askopenfilename()
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image_compress = cv2.imread(path)
        tf.app.flags.DEFINE_string("image_file", path, "")
        image_compress = cv2.cvtColor(image_compress, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image_compress = Image.fromarray(image_compress)
        srcw,srch = image_compress.size
        while(srcw > 650 or srcw > 650):
            srcw = int(srcw*(4/5))
            srch = int(srch*(4/5))
        image_compress = image_compress.resize( (srcw, srch), Image.BILINEAR )
    
    image_compress.save(path)
       

def select_model():
    options = {}
    options['initialdir'] = os.getcwd()
    path = filedialog.askopenfilename(**options)
    if len(path) > 0:
        tf.app.flags.DEFINE_string("model_file", path, "")


def Run_model():
    
    print("Image Processing...")
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

def close_window():
    window.destroy()


if __name__ == '__main__':
    window = Tk()
    window.title("ICG")
    panelA = None
    panelB = None
    btn_select_image = Button(window, text="Select an image", command=select_image)
    btn_select_image.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

    btn_select_model = Button(window, text="Select a model", command=select_model)
    btn_select_model.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

    button_run = Button (window, text = "Run", command = Run_model)
    button_run.pack(side="top", fill="both", padx="10", pady="10")

    button_exit = Button(window, text="Exit", command=close_window)
    button_exit.pack(side="top", fill="both", padx="10", pady="10")

    window.mainloop()



    #/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/platform/app.py
    #記得改回來
    # _sys.exit(main(_sys.argv[:1] + flags_passthrough))




    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run()
