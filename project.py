from unittest.mock import inplace

import streamlit as st
import numpy as np
import pandas as pd
import pickle

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
ss = pickle.load(open('scaler.pkl','rb'))

cols = ['Brand', 'RAM_MB', 'Internal_Memory_GB', 'Os_Type', 'Clock_Speed_GHz',
       'Battery_Capacity', 'Charging_Type', 'Is_5G', 'Display_Size_Inches',
       'Display_Width_px', 'Display_Height_px', 'Refresh_Rate_Hz']

st.title('Smartphone Price Predictor')

b_options = ['Xiaomi','Samsung','Vivo','Realme','Oppo',
             'Motorola','Oneplus','Poco','Iqoo','Nokia']
brand = st.selectbox('**Brand**',b_options)

ram_options = ['4 GB', '6 GB', '8 GB', '12 GB']
ram = st.selectbox('**RAM**',ram_options)
ram = int(ram.split(' ')[0])*1024

memory_options = ['32 GB', '64 GB', '128 GB', '256 GB', '512 GB']
memory = st.selectbox('**Internal Memory**',memory_options)
memory = int(memory.split(' ')[0])

os_options = ['Android v11', 'Android v12', 'Android v13']
os = st.radio('**Operating System**',os_options)

chip_options = ['Dimensity', 'Snapdragon', 'Helio']
chipset = st.radio('**Chipset**',chip_options)

#cs_options = ['1.8 GHz – 2.2 GHz', '2.4 GHz – 2.8 GHz', '2.8 GHz – 3.2 GHz']
cs = st.slider('**Clock Speed (GHz)**', min_value=1.8, max_value=3.2, step=0.1)

#battery_options = ['4,000 mAh – 5,000 mAh', '5,001 mAh – 5,500 mAh', '5,501 mAh – 6,000 mAh']
battery = st.slider('**Battery Capacity (mAh)**', min_value=4600, max_value=6000, step=200)

charge_options = ['Basic charging', 'Mid charging', 'Fast charging', 'Ultra fast charging']
charge = st.selectbox('**Charging Type**',charge_options)

conn_options = ['Yes', 'No']
conn = st.radio('**5G Connectivity**',conn_options)
if conn == 'Yes':
    conn = 1
else:
    conn = 0

#screen_options = ['6.00 " – 6.30 "', '6.31 " – 6.60 "', '6.61 " – 6.90 "']
screen_size = st.slider('**Screen Size (in)**', min_value=6.0, max_value=7.0, step=0.1)

res_options = ['HD+', 'Full HD+', '2K / QHD+']
res = st.radio('**Screen Resolution**',res_options)

res_type = {'HD+':(1600,720),
            'Full HD+':(2400,1080),
            '2K / QHD+':(3200,1440)}
res_h, res_w = res_type[res]

fc_options = ['5 MP – 8 MP', '10 MP – 16 MP', '20 MP – 32 MP', '42 MP – 50 MP']
front_camera = st.selectbox('**Front Camera**',fc_options)

rc_options = ['8 MP – 13 MP', '32 MP – 48 MP', '50 MP – 64 MP', '108 MP']
rear_camera = st.selectbox('**Rear Camera**',rc_options)

rr_options = ['90 Hz', '120 Hz', '144 Hz']
refresh_rate = st.radio('**Refresh Rate**', rr_options)
refresh_rate = refresh_rate.split(' ')[0]

query = np.array([brand,ram,memory,os,cs,battery,charge,conn,
                  screen_size,res_w,res_h,refresh_rate])
ask = pd.DataFrame([query], columns=cols)

st.write(''
        '')

if st.button('**Predict Price**'):
    scaled_pred = pipe.predict(ask)
    final_price = ss.inverse_transform(scaled_pred.reshape(-1, 1))

    st.markdown(f'**Rs {int(final_price[0][0])}**')