import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

st.title("Image Color Palette Extractor")

img=st.file_uploader('Upload any image and give number of colors to extract')


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
    st.image(img)

    img1=Image.open(img)

    data=np.array(img1)
    data_transformed=data.reshape((data.shape[0]*data.shape[1],data.shape[2]))

    num_clusters=st.number_input(label='Enter number of colors to extract',step=1,min_value=1,max_value=8)

    if num_clusters is not None:
        if st.button('Run🚀'):
            model=KMeans(n_clusters=num_clusters)
            clusters=model.fit_predict(data_transformed)

            color_list=['#{:02x}{:02x}{:02x}'.format(int(i[0]),int(i[1]),int(i[2])) for i in model.cluster_centers_]
            palette=np.zeros((1,num_clusters,data.shape[2]),dtype=np.uint8)
            palette[:,:]=model.cluster_centers_

            fig=plt.figure()
            plt.xticks([i for i in range(num_clusters)],labels=color_list)
            plt.yticks([])
            plt.imshow(palette)

            st.header('Dominant Colors')
            st.pyplot(fig)

