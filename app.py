import streamlit as st
import numpy as np
import pandas as pd
import os
import colorsys
import plotly.graph_objects as go
from xgboost import XGBRegressor

# -------------------------------------------------------------------
# 1. UI CONFIG & ADVANCED CSS (GLASSMORPHISM & GLOW)
# -------------------------------------------------------------------
st.set_page_config(page_title="AI UI Color Harmony", page_icon="🎨", layout="wide")

st.markdown("""
<style>
    /* Tổng thể nền Dark Mode sâu */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 20%, #1a1c23 0%, #0e1117 100%);
    }

    /* Sidebar: Nổi bật, không bị chìm */
    [data-testid="stSidebar"] {
        background: rgba(15, 18, 24, 0.95) !important;
        border-right: 1px solid rgba(0, 255, 204, 0.3);
        box-shadow: 10px 0 30px rgba(0,0,0,0.5);
    }

    /* Khối thông tin chuẩn UX/UI theo mẫu của Mỹ */
    .sidebar-info-block {
        margin-top: 30px;
        padding: 18px;
        background: rgba(0, 255, 204, 0.03);
        border-radius: 12px;
        border-left: 4px solid #1DF27D;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .info-label {
        color: #1DF27D;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 5px;
        margin-top: 15px;
    }
    .info-label:first-child { margin-top: 0; }
    .info-content {
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0px;
        line-height: 1.5;
    }
    .info-sub { color: #888; font-size: 0.8rem; font-weight: 400; }

    /* Card Glassmorphism (Hỗ trợ Native Container của Streamlit) */
    .glass-card, [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 20px !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 15px 50px rgba(0,0,0,0.6) !important;
        margin-bottom: 25px !important;
        transition: transform 0.3s ease !important;
    }
    .glass-card:hover, [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(0, 255, 204, 0.3) !important;
    }

    /* Tiêu đề Glow */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #fff, #1DF27D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        filter: drop-shadow(0 0 10px rgba(0, 255, 204, 0.3));
    }

    /* Score Glow Display */
    .score-container {
        text-align: center;
        padding: 20px;
    }
    .score-number {
        font-size: 8rem;
        font-weight: 900;
        background: linear-gradient(to bottom, #ffffff 30%, #1DF27D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 25px rgba(0, 255, 204, 0.7));
        line-height: 1;
    }

    /* Nhãn vai trò màu sắc */
    .role-tag {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CORE LOGIC (MODERN UI ENGINE)
# -------------------------------------------------------------------
def get_hue_distance(h1, h2):
    diff = abs(h1 - h2) % 360
    return diff if diff <= 180 else 360 - diff

def hex_to_hsv(h):
    h = h.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v

def hex_to_lab(h):
    h = h.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
    def f(n): return ((n+0.055)/1.055)**2.4 if n>0.04045 else n/12.92
    r,g,b = f(r), f(g), f(b)
    x = (r*0.4124 + g*0.3575 + b*0.1804)*100
    y = (r*0.2126 + g*0.7151 + b*0.0721)*100
    z = (r*0.0193 + g*0.1191 + b*0.9503)*100
    x/=95.047; y/=100; z/=108.883
    def p(n): return n**(1/3) if n>0.008856 else (903.3*n+16.16)/116
    return max(0, 116*p(y)-16), 500*(p(x)-p(y)), 200*(p(y)-p(z))

def extract_ui_features(hex_colors):
    """Trích xuất đúng 9 đặc trưng mà mô hình Modern UI yêu cầu"""
    hsv = [hex_to_hsv(c) for c in hex_colors]
    lab = [hex_to_lab(c) for c in hex_colors]
    
    # 1:Bg, 2:Surf, 3:Pri, 4:Sec, 5:Acc
    de_bg_surf = np.sqrt((lab[0][0]-lab[1][0])**2 + (lab[0][1]-lab[1][1])**2 + (lab[0][2]-lab[1][2])**2)
    de_p_s = np.sqrt((lab[2][0]-lab[3][0])**2 + (lab[2][1]-lab[3][1])**2 + (lab[2][2]-lab[3][2])**2)
    de_p_a = np.sqrt((lab[2][0]-lab[4][0])**2 + (lab[2][1]-lab[4][1])**2 + (lab[2][2]-lab[4][2])**2)
    
    return pd.DataFrame([{
        'contrast_bg_primary': abs(lab[0][0] - lab[2][0]),
        'contrast_bg_accent': abs(lab[0][0] - lab[4][0]),
        'de_bg_surf': de_bg_surf,
        'hue_dist_p_s': get_hue_distance(hsv[2][0], hsv[3][0]),
        'hue_dist_p_a': get_hue_distance(hsv[2][0], hsv[4][0]),
        'accent_vibrancy': hsv[4][1] * hsv[4][2],
        'bg_lightness': lab[0][0],
        'de_primary_sec': de_p_s,
        'de_primary_acc': de_p_a
    }])

# -------------------------------------------------------------------
# 3. SIDEBAR & CONTACT INFO
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#1DF27D; letter-spacing:2px; font-weight:800;'>DASHBOARD</h2>", unsafe_allow_html=True)
    st.info("Evaluation Rule: **Modern UI (Analogous + Accent)**")
    
    # Khối thông tin chuẩn UX/UI của Mỹ
    st.markdown("""
    <div class="sidebar-info-block">
    <div class="info-label" style="margin-top: 0;">Instructor / Advisor</div>
    <div style="color: white; font-size: 0.9rem; font-weight: 500; line-height: 1.2;">ThS. Vũ Minh Anh</div>
    <div style="color: #888; font-size: 0.8rem; margin-bottom: 12px; line-height: 1.2;">anhvm&commat;vnu&period;edu&period;vn</div>
    <div class="info-label">Lead Engineer</div>
    <div style="color: white; font-size: 0.9rem; font-weight: 500; line-height: 1.2;">Bùi Nhật Quang</div>
    <div style="color: #888; font-size: 0.8rem; margin-bottom: 12px; line-height: 1.2;">ID: 24023060 | VNU-UET</div>
    <div class="info-label">Personal Contact</div>
    <div style="color: white; font-size: 0.9rem; font-weight: 500; line-height: 1.2;">Mr. Q</div>
    <div style="color: #888; font-size: 0.8rem; line-height: 1.2;">hello&commat;misterq&period;net</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 4. MAIN LAYOUT
# -------------------------------------------------------------------
st.markdown("<div class='main-title'>AI EVALUATOR FOR UI COLOR HARMONY</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666; margin-top:-10px; font-weight:500;'>Consulting on standard color palettes for modern UI interface & Web Accessibility</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666; margin-top:-10px; font-weight:500;'>Tư vấn phối màu chuẩn giao diện UI hiện đại & Web Accessibility</p>", unsafe_allow_html=True)

# BƯỚC 1: CHỌN MÀU THEO VAI TRÒ
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("1. Palette Configuration (Cấu hình bảng màu)")
roles = ["Background", "Surface", "Primary", "Secondary", "Accent"]
default_hex = ["#0E1117", "#1A1C23", "#1DF27D", "#0099FF", "#FF3366"]
cols = st.columns(5)
selected_colors = []

for i in range(5):
    with cols[i]:
        st.markdown(f"<div class='role-tag'>{roles[i]}</div>", unsafe_allow_html=True)
        c = st.color_picker(f"Chọn {roles[i]}", key=f"c{i}", value=default_hex[i], label_visibility="collapsed")
        selected_colors.append(c)

# Hiển thị dải màu trực quan
swatch_html = "<div style='display:flex; gap:12px; margin-top:25px;'>"
for i, c in enumerate(selected_colors):
    l_val = hex_to_lab(c)[0]
    txt = 'black' if l_val > 55 else 'white'
    swatch_html += f"""
    <div style='flex:1; height:100px; border-radius:16px; background:{c}; 
                border:1px solid rgba(255,255,255,0.1); display:flex; flex-direction:column; 
                align-items:center; justify-content:center; font-weight:700; color:{txt};'>
        <span style='font-size:0.8rem;'>{c.upper()}</span>
    </div>"""
swatch_html += "</div>"
st.markdown(swatch_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# BƯỚC 2 & 3: PHÂN TÍCH & KẾT QUẢ
col_analysis, col_score = st.columns([1.3, 1])

with col_analysis:
            # Sử dụng st.container(border=True) để bọc biểu đồ vào khung Glassmorphism thực thụ
            with st.container(border=True):
                st.subheader("2. Palette Health Check")
                feats = extract_ui_features(selected_colors)
                
                # Radar Chart
                categories = ['Accessibility', 'Harmony', 'Diversity', 'Vibrancy', 'Layering']
                values = [
                    min(100, feats['contrast_bg_primary'][0] * 1.5), 
                    max(0, 100 - (feats['hue_dist_p_s'][0] / 60 * 100)), 
                    min(100, feats['de_primary_acc'][0] * 2), 
                    min(100, feats['accent_vibrancy'][0] * 100), 
                    min(100, feats['de_bg_surf'][0] * 5) 
                ]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]], theta=categories + [categories[0]],
                    fill='toself', fillcolor='rgba(0, 255, 204, 0.25)',
                    line=dict(color='#1DF27D', width=2.5)
                ))
                
                # Cải thiện thẩm mỹ: xóa nền trắng, làm line sắc nét hơn
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)', # Xóa nền tròn màu trắng
                        radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(0, 255, 204, 0.15)", linecolor="rgba(0,0,0,0)"),
                        angularaxis=dict(gridcolor="rgba(0, 255, 204, 0.2)", linecolor="rgba(0,0,0,0)", tickfont=dict(color="white", size=12))
                    ),
                    showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=30, b=30), height=380
                )
                # Bổ sung theme=None để loại bỏ nền xám (thẻ bị chồng) của Streamlit
                st.plotly_chart(fig_radar, use_container_width=True, theme=None, config={'displayModeBar': False})

with col_score:
    # Sử dụng st.container(border=True) cho phần điểm số và kết quả
    with st.container(border=True):
        st.subheader("3. Harmony Result (Kết quả)")
        
        if st.button("EVALUATE (CHẤM ĐIỂM)", type="primary", use_container_width=True):
            m_path = os.path.join("models", "saved_models", "model_modern_ui.json")
            if os.path.exists(m_path):
                try:
                    model = XGBRegressor()
                    model.load_model(m_path)
                    base_score = float(model.predict(feats)[0])
                    
                    # GUARDRAIL: Kiểm tra tính đa dạng (Diversity)
                    lab = [hex_to_lab(c) for c in selected_colors]
                    d_p_s = np.sqrt((lab[2][0]-lab[3][0])**2 + (lab[2][1]-lab[3][1])**2 + (lab[2][2]-lab[3][2])**2)
                    d_p_a = np.sqrt((lab[2][0]-lab[4][0])**2 + (lab[2][1]-lab[4][1])**2 + (lab[2][2]-lab[4][2])**2)
                    min_de = min(d_p_s, d_p_a)
                    
                    penalty = 0
                    if min_de < 12:
                        penalty = 3.5 * (1 - (min_de/12))
                    
                    final_score = max(1.0, min(10.0, base_score - penalty))

                    # Gauge Chart - Cải thiện thẩm mỹ UI
                    
                    if final_score < 6.5:       # Thay đổi màu của gauge chart tùy theo điểm số
                        gauge_color = "#FF3366"
                    elif final_score < 8.5:
                        gauge_color = "#FFCC00"
                    else:
                        gauge_color = "#1DF27D"
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number", value = final_score,                       
                        number = {'font': {'color': gauge_color, 'size': 60, 'family': "Arial Black"}, 'valueformat': ".1f"},
                        gauge = {'axis': {'range': [1, 10], 'tickwidth': 2, 'tickcolor': "#1DF27D", 'ticklen': 8},
                                 'bar': {'color': gauge_color, 'thickness': 0.8},
                                 'bgcolor': "rgba(0,0,0,0)",
                                 'borderwidth': 1,
                                 'bordercolor': "rgba(0, 255, 204, 0.2)",
                                 'steps': [{'range': [1, 6.5], 'color': 'rgba(255, 51, 102, 0.15)'},  # Đỏ nhạt
                                           {'range': [6.5, 8.5], 'color': 'rgba(255, 204, 0, 0.15)'},  # Vàng nhạt
                                           {'range': [8.5, 10], 'color': 'rgba(0, 255, 204, 0.15)'}]} # Cyan nhạt
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white"}, margin=dict(l=30, r=30, t=30, b=10), height=250
                    )
                    # Bổ sung theme=None để loại bỏ nền xám (thẻ bị chồng) của Streamlit
                    st.plotly_chart(fig_gauge, use_container_width=True, theme=None, config={'displayModeBar': False})
                    
                    # Hiển thị các Impact Factors ngay trong container
                    st.markdown(f"<p style='color:#1DF27D; font-size:0.75rem; font-weight:bold; margin-top:10px;'>IMPACT FACTORS (YẾU TỐ ẢNH HƯỞNG)</p>", unsafe_allow_html=True)
                    st.progress(min(1.0, feats['contrast_bg_primary'][0]/70), text=f"Text Contrast (Độ tương phản văn bản): **{feats['contrast_bg_primary'][0]:.1f}**")
                    st.progress(min(1.0, feats['contrast_bg_accent'][0]/70), text=f"CTA Contrast (Độ tương phản nút CTA): **{feats['contrast_bg_accent'][0]:.1f}**")
                    
                    if penalty > 0: 
                        st.error("Cảnh báo: Màu sắc bị trùng lặp!")
                except Exception as e: 
                    st.error(f"Lỗi nạp mô hình: {e}")
            else: 
                st.error("Model 'model_modern_ui.json' không tồn tại!")
        else:
            # Thông báo khi chưa nhấn nút phân tích
            st.markdown("<div style='margin-top:100px; margin-bottom:100px; text-align:center; color:#444; font-weight:500;'>Press the button to evaluate (Nhấn nút để phân tích)...</div>", unsafe_allow_html=True)