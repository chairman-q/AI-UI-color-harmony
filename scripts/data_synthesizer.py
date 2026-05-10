import os
import numpy as np
import pandas as pd
import colorsys

RAW_DATA_PATH = os.path.join("data", "raw")
os.makedirs(RAW_DATA_PATH, exist_ok=True)

# ==========================================
# 1. CÁC HÀM TOÁN HỌC MÀU SẮC CHUẨN XÁC
# ==========================================
def hsv_to_rgb(h, s, v):
    # h trong khoảng [0, 360], s và v [0, 1]
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return r, g, b

def rgb_to_lab(rgb):
    r, g, b = rgb
    def pivot_rgb(n): return ((n + 0.055) / 1.055) ** 2.4 if n > 0.04045 else n / 12.92
    r, g, b = pivot_rgb(r), pivot_rgb(g), pivot_rgb(b)
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) * 100
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) * 100
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) * 100
    x /= 95.047; y /= 100.000; z /= 108.883
    def pivot_xyz(n): return n ** (1/3) if n > 0.008856 else (903.3 * n + 16.16) / 116
    x, y, z = pivot_xyz(x), pivot_xyz(y), pivot_xyz(z)
    return max(0, 116 * y - 16), 500 * (x - y), 200 * (y - z)

def get_luminance(rgb):
    """Tính độ chói tương đối (Relative Luminance) theo chuẩn WCAG"""
    def f(c): return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = [f(v) for v in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_contrast_ratio(rgb1, rgb2):
    """Tính tỷ lệ tương phản giữa 2 màu (1:1 đến 21:1)"""
    l1 = get_luminance(rgb1)
    l2 = get_luminance(rgb2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

def delta_e(lab1, lab2):
    """Tính khoảng cách thị giác Delta E76"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))

# ==========================================
# 2. HÀM CHẤM ĐIỂM UI/UX CHUYÊN SÂU
# ==========================================
def calculate_modern_ui_score(hsv_p, rgb_p, lab_p):
    score = 10.0
    bg, surface, primary, secondary, accent = rgb_p
    
    # Tiêu chí 1: Độ tương phản nền và chữ (Khả năng đọc - Accessibility)
    # Primary và Accent thường chứa text hoặc icon đặt trên Background
    cr_primary = get_contrast_ratio(bg, primary)
    cr_accent = get_contrast_ratio(bg, accent)
    
    # Phạt nặng nếu vi phạm chuẩn WCAG AA (Contrast < 4.5)
    if cr_primary < 4.5: score -= (4.5 - cr_primary) * 1.5
    if cr_accent < 4.5: score -= (4.5 - cr_accent) * 1.5

    # Tiêu chí 2: Background và Surface phải phân tách nhẹ nhàng
    de_bg_surf = delta_e(lab_p[0], lab_p[1])
    if de_bg_surf < 3: score -= 2.0  # Quá giống nhau, không nổi card lên được
    if de_bg_surf > 15: score -= 1.5 # Khác biệt quá lớn, phá vỡ sự liền mạch của nền

    # Tiêu chí 3: Primary và Secondary phải có tính Analogous (Tương đồng)
    h_primary, h_secondary = hsv_p[2][0], hsv_p[3][0]
    hue_dist_p_s = min(abs(h_primary - h_secondary), 360 - abs(h_primary - h_secondary))
    if hue_dist_p_s > 60: score -= 2.0 # Nếu lệch xa quá không còn là Analogous nữa

    # Tiêu chí 4: Accent phải thực sự nổi bật (Pop)
    h_accent = hsv_p[4][0]
    hue_dist_p_a = min(abs(h_primary - h_accent), 360 - abs(h_primary - h_accent))
    # Accent nên khác biệt màu Primary ít nhất 90 độ, hoặc có Saturation rất cao
    if hue_dist_p_a < 60 and hsv_p[4][1] < 0.8:
        score -= 2.5 # Accent chìm nghỉm, không hoàn thành nhiệm vụ CTA

    # Tiêu chí 5: Tránh "Dead Palette" (Các màu chính bị trùng nhau)
    if delta_e(lab_p[2], lab_p[3]) < 10: score -= 2.0
    if delta_e(lab_p[2], lab_p[4]) < 15: score -= 3.0

    # Tiêm nhiễu chủ quan (Human UI/UX Subjectivity)
    human_noise = np.random.normal(0, 0.4)
    score += human_noise
    
    return max(1.0, min(10.0, score))

# ==========================================
# 3. TRÌNH KIẾN TẠO DỮ LIỆU ĐA DẠNG
# ==========================================
def generate_modern_ui_dataset(num_samples=5000):
    rows = []
    np.random.seed(42)
    
    for _ in range(num_samples):
        theme = np.random.choice(['light', 'dark'])
        brand_hue = np.random.uniform(0, 360)
        
        # --- Tạo cấu trúc Base (Palette Đẹp) ---
        if theme == 'light':
            bg = (brand_hue, np.random.uniform(0, 0.05), np.random.uniform(0.95, 1.0))
            surf = (brand_hue, np.random.uniform(0, 0.08), np.random.uniform(0.90, 0.98))
        else: # Dark mode
            bg = (brand_hue, np.random.uniform(0.1, 0.3), np.random.uniform(0.05, 0.15))
            surf = (brand_hue, np.random.uniform(0.1, 0.3), np.random.uniform(0.12, 0.22))
            
        # Primary & Secondary (Tone màu thương hiệu)
        primary = (brand_hue, np.random.uniform(0.5, 0.9), np.random.uniform(0.6, 0.9))
        secondary_hue = (brand_hue + np.random.choice([30, -30, 45, -45])) % 360
        secondary = (secondary_hue, np.random.uniform(0.4, 0.8), np.random.uniform(0.5, 0.8))
        
        # Accent (Điểm nhấn - Complementary hoặc Triadic)
        accent_hue = (brand_hue + np.random.choice([180, 150, 210])) % 360
        accent = (accent_hue, np.random.uniform(0.8, 1.0), np.random.uniform(0.8, 1.0))
        
        hsv_p = [bg, surf, primary, secondary, accent]

        # --- HARD NEGATIVES (Bẫy AI bằng các lỗi thiết kế phổ biến) ---
        if np.random.rand() < 0.4:
            trap_type = np.random.choice(['bad_contrast', 'dead_accent', 'neon_bg', 'muddy_colors'])
            if trap_type == 'bad_contrast':
                # Làm cho Primary có độ chói y hệt Background
                hsv_p[2] = (brand_hue, primary[1], bg[2]) 
            elif trap_type == 'dead_accent':
                # Accent bị xỉn màu và trùng với Primary
                hsv_p[4] = (brand_hue, 0.2, 0.5)
            elif trap_type == 'neon_bg':
                # Đau mắt: Background chói lóa
                hsv_p[0] = (np.random.uniform(0,360), 0.9, 0.9)
            elif trap_type == 'muddy_colors':
                # Mọi màu đều xám xịt
                hsv_p = [(h, 0.1, 0.4) for h, s, v in hsv_p]

        # Xử lý tính toán
        rgb_p = [hsv_to_rgb(*p) for p in hsv_p]
        lab_p = [rgb_to_lab(p) for p in rgb_p]
        
        score = calculate_modern_ui_score(hsv_p, rgb_p, lab_p)

        # Lưu dữ liệu
        row = {'num_colors': 5}
        for i in range(5):
            row[f'c{i+1}_h'] = round(hsv_p[i][0], 2)
            row[f'c{i+1}_s'] = round(hsv_p[i][1], 3)
            row[f'c{i+1}_v'] = round(hsv_p[i][2], 3)
            row[f'c{i+1}_L'] = round(lab_p[i][0], 2)
            row[f'c{i+1}_a'] = round(lab_p[i][1], 2)
            row[f'c{i+1}_b'] = round(lab_p[i][2], 2)
        row['target_score'] = round(score, 2)
        rows.append(row)
        
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("[*] Đang kiến tạo dữ liệu Modern UI (5000 mẫu)...")
    df = generate_modern_ui_dataset(5000)
    file_path = os.path.join(RAW_DATA_PATH, "data_modern_ui.csv")
    df.to_csv(file_path, index=False)
    print(f"[+] Hoàn tất! Đã lưu tại: {file_path}")
    print("[+] Phân phối điểm số:")
    print(df['target_score'].describe())