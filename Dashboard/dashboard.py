import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Đặt chế độ wide mode cho Streamlit
st.set_page_config(layout="wide")

# Hàm vẽ biểu đồ 1: Lượng mưa tại các địa điểm
def plot_precipitation(df):
    plt.figure(figsize=(12, 7.5))  # Tinh chỉnh kích thước phù hợp với màn hình
    palette = sns.color_palette("husl", len(df['location'].unique()))
    sns.boxplot(data=df, x='location', y='precipitation_sum', palette=palette, hue='location')
    plt.xticks(rotation=45, ha='right')
    plt.title('Lượng mưa tại các địa điểm')
    plt.ylabel('Lượng mưa')
    plt.xlabel('Địa điểm')
    plt.tight_layout()

    st.pyplot(plt)
    plt.clf()

# Hàm vẽ biểu đồ 2: Số lượng các mã thời tiết
def plot_weather_code(df):
    weather_code_map = {
    0: 'Clear sky',
    1: 'Mainly clear',
    2: 'Partly cloudy',
    3: 'Overcast',
    51: 'Drizzle',
    53: 'Heavy drizzle',
    55: 'Freezing drizzle',
    61: 'Showers of rain',
    63: 'Heavy showers of rain',
    65: 'Showers of snow',
    }   

    plt.figure(figsize=(12, 4))

    # Chuẩn bị dữ liệu
    weather_counts = df['weather_code'].value_counts()  # Đếm số lượng từng mã thời tiết
    weather_labels = [f"{code}: {weather_code_map[code]}" for code in weather_counts.index]  # Tạo nhãn mô tả chi tiết

    # Vẽ biểu đồ tròn
    colors = sns.color_palette('viridis', len(weather_counts))[::-1]  # Đảo ngược bảng màu
    plt.pie(
        weather_counts,
        labels=weather_counts.index,  # Hiển thị mã thời tiết trong biểu đồ
        colors=colors,
        autopct='%1.1f%%',  # Hiển thị phần trăm
        startangle=90,  # Bắt đầu từ góc 90 độ
    )

    # Thêm chú thích (custom legend)
    plt.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors[i])
            for i in range(len(weather_counts))
        ],
        labels=weather_labels,
        title='Weather Codes',
        bbox_to_anchor=(1.05, 1),  # Đặt chú thích ở bên phải biểu đồ
        loc='upper left',
        frameon=False
    )

    # Cài đặt tiêu đề
    plt.title('Tỷ lệ các mã thời tiết (weather_code)')
    plt.tight_layout()
    st.pyplot(plt)  # Hiển
    

# Hàm vẽ biểu đồ 3: Hướng gió chủ đạo
def plot_wind_direction(df):
    wind_directions = df['wind_direction_10m_dominant'].dropna()
    angles = np.deg2rad(wind_directions)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.hist(angles, bins=36, color='green', edgecolor='black')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    directions = ['0° - N', '45° - NE', '90° - E', '135° - SE', '180° - S', '225° - SW', '270° - W', '315° - NW']
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ax.set_xticks(angles)
    ax.set_xticklabels(directions)
    plt.title('Hướng gió chủ đạo')
    st.pyplot(plt)
    plt.clf()

# Hàm vẽ biểu đồ 4: Tỷ lệ ngày mưa và không mưa theo từng địa điểm
def plot_rain_by_location(df, locations):
    rain_by_location = df.groupby('location')['RainToday'].value_counts(normalize=True).unstack().reindex(locations)
    rain_by_location.plot(kind='bar', figsize=(12, 6.3), color=['skyblue', 'orange'], edgecolor='black')
    plt.title('Tỷ lệ ngày không mưa và có mưa theo từng location')
    plt.xlabel('Location')
    plt.ylabel('Tỷ lệ')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()

# Hàm vẽ biểu đồ 5: Tổng lượng mưa theo khu vực
def plot_precipitation_location(df, locations):
    df.groupby('location')['precipitation_sum'].sum().reindex(locations).plot(kind='bar', figsize=(12, 5.95), color='skyblue', edgecolor='black')
    plt.title('Tổng lượng mưa theo khu vực')
    plt.xlabel('Khu vực')
    plt.ylabel('Tổng lượng mưa (mm)')
    st.pyplot(plt)
    plt.clf()

# Hàm vẽ biểu đồ 6: Mối quan hệ giữa thời gian chiếu sáng và nhiệt độ trung bình
def plot_sun_duration_vs_temperature(df):
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='daylight_duration', y='temperature_2m_mean', data=df, color='dodgerblue', s=100, alpha=0.6, edgecolor='black')
    plt.title('Mối quan hệ giữa thời gian chiếu sáng và nhiệt độ trung bình (Daylight Duration)')
    plt.xlabel('Thời gian chiếu sáng (giờ)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='sunshine_duration', y='temperature_2m_mean', data=df, color='orange', s=100, alpha=0.6, edgecolor='black')
    plt.title('Mối quan hệ giữa thời gian chiếu sáng và nhiệt độ trung bình (Sunshine Duration)')
    plt.xlabel('Thời gian chiếu sáng (giờ)')
    plt.ylabel('Nhiệt độ trung bình (°C)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    st.pyplot(plt)
    plt.clf()

# Hàm vẽ biểu đồ 7: Nhiệt độ trung bình theo ngày có mưa và không mưa
def plot_temperature_rain(df):
    plt.figure(figsize=(12, 12.4))
    sns.boxplot(x='RainToday', y='temperature_2m_mean', data=df, palette='coolwarm', hue='RainToday', dodge=False)
    plt.title('Nhiệt độ trung bình theo ngày có mưa và không mưa', fontsize=20)
    plt.xlabel('Có mưa',fontsize=20)
    plt.ylabel('Nhiệt độ trung bình (°C)',fontsize=20)
    st.pyplot(plt)
    plt.clf()

# Gọi các hàm vẽ biểu đồ trong Streamlit

# Giả sửf dữ liệu của bạn
# df = ...  # Tải dữ liệu của bạn vào đây
# locations = ...  # Định nghĩa danh sách locations của bạn ở đây
import pandas as pd
df = pd.read_csv('../Dataset - Giai đoạn 1/HCMCity_weather.csv')
locations = pd.read_csv('../Dataset gốc/location_coordinates.csv')['Location'].values
df['RainToday'] = df['precipitation_sum'].apply(lambda x: 1 if x > 1 else 0)

# Tạo container chứa các biểu đồ
with st.container():
    # Row 1: Hai biểu đồ
    col1, col2 = st.columns(2, gap="small")
    with col1:
        plot_precipitation(df)
    with col2:
        plot_weather_code(df)

    # Row 2: Hai biểu đồ với tỷ lệ cột khác nhau
    col3, col4 = st.columns([1, 3])
    with col3:
        plot_wind_direction(df)
    with col4:
        col5, col6 = st.columns(2)
        with col5:
            plot_rain_by_location(df, locations)
        with col6:
            plot_precipitation_location(df, locations)

    # Row 3: Hai biểu đồ với tỷ lệ cột khác nhau
    col7, col8 = st.columns([3, 1])
    with col7:
        plot_sun_duration_vs_temperature(df)
    with col8:
        plot_temperature_rain(df)

# Để đảm bảo kích thước container phù hợp với màn hình, bạn có thể thêm CSS để điều chỉnh
st.markdown("""
    <style>
        .container {
            max-width: 100vw;  /* Đặt chiều rộng tối đa của container */
            overflow: hidden;  /* Ẩn thanh cuộn ngang nếu có */
        }
    </style>
""", unsafe_allow_html=True)