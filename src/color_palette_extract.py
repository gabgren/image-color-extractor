import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt 

st.title("Image Color Palette Extractor")

img=st.file_uploader('Upload any image (png, jpg or jpeg) and give number of colors to extract',type=['png', 'jpg','jpeg'])


with st.sidebar:
    st.image("src/icons/2867978.png")
    st.header("Image Color Palette Extractor")
    st.text("""Image dominant color palette 
extractor using 
K-Means Clustering.
Based on number of colors given, 
clusters are created.
    """)

if img is not None :
    st.write(":green[Image uploaded successfully!]")
    
    img1=Image.open(img)

    # make sure its a 3 channel image (ignore alpha channel)
    img1=img1.convert('RGB')

    # blur image
    img1=img1.filter(ImageFilter.GaussianBlur(10))

    st.image(img1)

    # find the average pixel color
    img1 = np.array(img1)

    # black out all pixels above and below a certain luma value
    img1[img1.mean(axis=2) < 255*.25] = 0
    img1[img1.mean(axis=2) > 255*.70] = 0

    print(img1.shape)
    # only keep pixels that are not black, but keep the shape
    img1 = img1[np.any(img1 != 0, axis=2)]
    num_pixels = img1.shape[0]
    side_length = int(np.sqrt(num_pixels))

    # sort
    img1 = np.sort(img1, axis=0)

    # Reshape the array to a square, discarding extra pixels if necessary
    img1 = img1[:side_length*side_length].reshape(side_length, side_length, 3)

    print(img1.shape)

    st.image(img1)


    data_transformed=img1.reshape((img1.shape[0]*img1.shape[1],img1.shape[2]))

    num_clusters=st.number_input(label='Enter number of colors to extract',step=1,min_value=1,max_value=8)

    if num_clusters is not None:
        if st.button('Run🚀'):
            model=KMeans(n_clusters=num_clusters)
            model.fit_predict(data_transformed)

            cluster_color = model.cluster_centers_[0]
            cluster_color = [int(i) for i in cluster_color]  
            print(cluster_color)      

            color_list=['#{:02x}{:02x}{:02x}'.format(int(i[0]),int(i[1]),int(i[2])) for i in model.cluster_centers_]

            print("CLUSTERS", color_list)

            palette=np.zeros((1,num_clusters,img1.shape[2]),dtype=np.uint8)
            palette[:,:]=model.cluster_centers_

            fig=plt.figure()
            plt.xticks([i for i in range(num_clusters)],labels=color_list)
            plt.yticks([])
            plt.imshow(palette)

            st.header('Dominant Colors')
            st.pyplot(fig)

