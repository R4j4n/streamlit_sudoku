import cv2
import webbrowser
import numpy as np 
from PIL import Image
import streamlit as st
from utils import *
from backtracking import * 
from tensorflow.keras.models import load_model


def download_img():

    st.title("Here are some images the model haven't seen")
    st.text("But feel free to upload your image.") 
    if st.button("GO BACK 🏃"):
        main()
        begin()
    st.image('static/level3.jpg',caption = "Difficult 🥵")
    st.image('static/level2.jpg' ,caption = "Moderate 😬")
    st.image('static/level1.png' ,caption = "Easy 🥱 ")

def main():
    st.title("🔥SUDOKU SOLVER🔥") 
    st.write("Solving Sudoku form images using ImageProcessing , DeepLearning and Backtracking ")
    st.write( "How to use it ? 🤔 ")
    st.write("Take photo of Sudoku gird from newspaper,book or whatever you wish and upload it below to solve it or use the images given below for testing purpose.You don't need to worry about prespective of the gird.")
    st.image('static/AI.png',width = 600)
    st.sidebar.title("how it works?🤔")
    st.sidebar.text("1.First the input image is converted into threshold image.")
    st.sidebar.text("2.Find contours within image and take the biggest square contour from the image.")
    st.sidebar.text("3.Get perspective transform biggest square contour and divide the wrapped img into 81 cell.")
    st.sidebar.text("4.For each cell find if it contains digit or not. if it contains digit then process the cell and center the digit in the cell and do other necessary preprocessing.")
    st.sidebar.text("5.Using NeuralNetworks helps in predicting the digit in the cells.")
    st.sidebar.text("6.Using Backtracking solve the sudoku and display the result")
    st.sidebar.title("FOLLOW ME :sunglasses: :")
    url = 'https://www.facebook.com/R1j1n'
    if st.sidebar.button('Facebook'):
        webbrowser.open_new_tab(url)
    st.title("Download images to test:")
    if st.button(" 👉HERE👈"):
        download_img()
    st.title("If you got the image lets begin🥳.")
    st.info( " 👇 UPLOAD YOUR SUDOKU IMAGE HERE 👇")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    image_file = st.file_uploader("upload",type=['jpeg', 'png', 'jpg', 'webp'])
    if image_file is not None:

        image = Image.open(image_file)


        if st.button("🔧🔧SOLVE🔧🔧"):
            
            st.warning(" ❗️ please consider the following exmaples ❗️ ")
            st.image('static/Red.jpg' ,width = 100,caption = "Red digit indicates digits detected from board")
            st.image('static/Blue.jpg',width = 100 ,caption = "Blue digit indicates the digits to be fill on board")
            # ACTUAL SOLVING 
            with st.spinner("Doing the hardowrk🔨🔨🔨"):
                model = load_model('digit.h5')
                img = np.array(image.convert('RGB'))
                p_img = thresholding(img)
                wrapped_img = wrap_img(p_img)
                tiles = split_img(wrapped_img)
                digits_highlight = []
                predictions = []
                for tile in tiles:

                    crop = crop_tile(tile)
                    count_white_pixles = cv2.countNonZero(crop)
                    if count_white_pixles > 150 :
                        centered_Text = get_text_center(crop)
                        img = cv2.resize(centered_Text,(28,28))
                        img = img.reshape((1,28,28,1))
                        x = img.astype('float32')/255
                        result = model.predict(x)
                        index = np.argmax(result) + 1
                        predictions.append(index)
                        digits_highlight.append(index)
                        
                    else:
                        x = 0
                        predictions.append(x)
                        digits_highlight.append(x)

                board = np.array(predictions).reshape((9, 9))
                solver = BackTracing(board)
                solver.solve()
                final = solver.bo
                board_h = np.array(digits_highlight).reshape((9, 9))
                different = beautiful_graphics_version2(final,board_h)
                img = cv2.cvtColor(different, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                st.title("Here is the solution to the image :")

                st.image(im_pil,width = 800 ,caption = "Final Result 😁 ") 
                st.balloons()
        
                                
if __name__ == "__main__":
    main()