import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Set the page title
st.set_page_config(layout="wide",page_title="Ho Chi Minh City Rain Forecast", page_icon=":alembic:")

# Sidebar
image_path = "https://bna.1cdn.vn/2024/08/03/i.ytimg.com-vi-jgiywzpss3u-_maxresdefault.jpg"
st.sidebar.image(image_path, use_column_width=True)
st.sidebar.title('`Ho Chi Minh City Rain Forecast`')
st.sidebar.write("Đồ án môn học: Phân tích dữ liệu - IE224.P11.CNCL")
st.sidebar.write("Hồ Quang Lâm - 21521049 - HTTT")
st.sidebar.write("Lê Minh Chánh - 21521882 - HTTT")
st.sidebar.write("Trần Thị Thanh Trúc - 21522722 - MTT&TTDL")


# Main page
st.header(':kiwifruit: Ho Chi Minh City Rain Forecast')
st.markdown("<hr>", unsafe_allow_html=True)


df = pd.read_csv('../Dataset - Hoàn chỉnh/HCMCity_weather.csv')
locations = df['location'].unique()

# Train model
def train_model(df, locations):
    models = {}
    scalers = {}
    for location in locations:
        df_location = df[df['location'] == location]

        x = df_location.drop(columns=['RainTomorrow', 'location'])
        y = df_location['RainTomorrow']

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        model = XGBClassifier(
            n_estimators=100,        # Số lượng cây
            learning_rate=0.01,      # Tốc độ học nhỏ hơn
            max_depth=3,             # Độ sâu tối đa của cây
            subsample=0.8,           # Tỷ lệ mẫu được chọn ngẫu nhiên từ tập huấn luyện
            colsample_bytree=0.8,    # Tỷ lệ đặc trưng được chọn ngẫu nhiên cho mỗi cây
            reg_alpha=0.1,           # Regularization L1
            reg_lambda=1,            # Regularization L2
            random_state=42
        )
        model.fit(x, y)
        scalers[location] = scaler
        models[location] = model

    return models, scalers

# tạo 22 model cho 22 location
models, scalers = train_model(df, locations)

# Địa điểm và áp suất
st.header("Thông tin địa điểm và áp suất")
location = st.selectbox("Địa điểm", locations)
col1, col2 = st.columns(2)
with col1:
    relative_humidity_2m = st.number_input("Độ ẩm tương đối ở(%)", min_value=0.0, max_value=100.0, step=0.1)
with col2:
    surface_pressure = st.number_input("Áp suất mặt đất (hPa)", min_value=900, max_value=1100, step=1)

# Nhóm thông số nhiệt độ
st.header("Thông số thời tiết")
col1, col2 = st.columns(2)
weather_code_options = {
    0: "Trời quang - 0",
    1: "Ít mây - 1",
    2: "Nhiều mây - 2",
    3: "Mây u ám - 3",
    51: "Mưa phùn nhẹ - 51",
    53: "Mưa nhỏ gián đoạn - 53",
    55: "Mưa phùn nặng - 55",
    61: "Mưa rào nhẹ - 61",
    63: "Mưa to gián đoạn - 63",
    65: "Mưa rào lớn - 65",
}

with col1: 
    cloud_cover = st.number_input("Độ che phủ mây (%)", min_value=0.0, max_value=100.0, step=0.1)
    weather_code = st.selectbox(
        "Mã thời tiết",
        options=list(weather_code_options.keys()),  # Các mã thời tiết
        format_func=lambda x: weather_code_options[x]  # Hiển thị mô tả
    )
with col2: 
    temperature_min = st.number_input("Nhiệt độ tối thiểu (°C)", min_value=0.0, max_value=50.0, step=0.1)
    temperature_max = st.number_input("Nhiệt độ tối đa (°C)", min_value=0.0, max_value=50.0, step=0.1)


st.header("Thông tin về ánh sáng và mưa")
col1, col2 = st.columns(2)
with col1:
    daylight_duration = st.number_input("Thời gian có ánh sáng ban ngày (giây)", min_value=0, max_value=86400, step=1)
    sunshine_duration = st.number_input("Thời gian có nắng (giây)", min_value=0, max_value=86400, step=1)
with col2:
    precipitation_sum = st.number_input("Tổng lượng mưa (mm)", min_value=0.0, max_value=500.0, step=0.1)
    precipitation_hours = st.number_input("Thời gian mưa (giờ)", min_value=0, max_value=24, step=1)

# Thông tin về gió
st.header("Thông tin về gió")
col1, col2 = st.columns(2)
with col1:
    wind_speed_max = st.number_input("Tốc độ gió tối đa (km/h)", min_value=0.0, max_value=200.0, step=0.1)
with col2:
    wind_direction = st.selectbox("Hướng gió", options=["Bắc (0 - 22.5)", "Đông Bắc 22.5 - 67.5)", "Đông (67.5 - 112.5)", "Đông Nam (112.5 - 157.5)", "Nam (157.5 - 202.5)", "Tây Nam (202.5 - 247.5)", "Tây (247.5 - 292.5)", "Tây Bắc (292.5 - 337.5)"])

    # Ánh xạ hướng gió sang số độ
    direction_to_angle = {
        "Bắc (0 - 22.5)": 0,
        "Đông Bắc (22.5 - 67.5)": 45,
        "Đông (67.5 - 112.5)": 90,
        "Đông Nam (112.5 - 157.5)": 135,
        "Nam (157.5 - 202.5)": 180,
        "Tây Nam (202.5 - 247.5)": 225,
        "Tây (247.5 - 292.5)": 270,
        "Tây Bắc (292.5 - 337.5)": 315
    }

    wind_direction = direction_to_angle[wind_direction]
et0_fao_evapotranspiration = st.number_input(
    "Chỉ số bốc hơi FAO (mm/ngày)", min_value=0.0, step=0.1
)


# Tạo nút dự đoán
if st.button("Dự báo"):
    data = {
        'relative_humidity_2m': relative_humidity_2m,
        'surface_pressure': surface_pressure,
        'cloud_cover': cloud_cover,
        'weather_code': weather_code,
        'temperature_2m_max': temperature_max,
        'temperature_2m_min': temperature_min,
        'daylight_duration': daylight_duration,
        'sunshine_duration': sunshine_duration,
        'precipitation_sum': precipitation_sum,
        'precipitation_hours': precipitation_hours,
        'wind_speed_10m_max': wind_speed_max,
        'wind_direction_10m_dominant': wind_direction,
        'et0_fao_evapotranspiration': et0_fao_evapotranspiration
    }

    x = pd.DataFrame(data, index=[0])
    model = models.get(location, None)
    scaler = scalers.get(location, None)

    x = scaler.transform(x)
    prediction = model.predict(x)[0]

    if prediction.astype(bool) == True:
        st.error("Dự báo mưa vào ngày mai")
    else:
        st.success("Dự báo không mưa vào ngày mai")


    