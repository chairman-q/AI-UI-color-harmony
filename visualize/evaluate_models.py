import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Thiết lập thẩm mỹ cho biểu đồ chuẩn báo cáo
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

# Thiết lập đường dẫn
RAW_DATA_PATH = os.path.join("data", "raw")
MODEL_SAVE_PATH = os.path.join("models", "saved_models")
VISUALIZE_PATH = "visualize"
os.makedirs(VISUALIZE_PATH, exist_ok=True)

# -------------------------------------------------------------------
# 1. ĐỒNG BỘ LOGIC TRÍCH XUẤT ĐẶC TRƯNG (Giống hệt file train)
# -------------------------------------------------------------------
def get_hue_distance(h1, h2):
    diff = abs(h1 - h2) % 360
    return diff if diff <= 180 else 360 - diff

def engineer_ui_features(df):
    features = []
    for _, row in df.iterrows():
        de_bg_surf = np.sqrt((row['c1_L']-row['c2_L'])**2 + (row['c1_a']-row['c2_a'])**2 + (row['c1_b']-row['c2_b'])**2)
        de_p_s = np.sqrt((row['c3_L']-row['c4_L'])**2 + (row['c3_a']-row['c4_a'])**2 + (row['c3_b']-row['c4_b'])**2)
        de_p_a = np.sqrt((row['c3_L']-row['c5_L'])**2 + (row['c3_a']-row['c5_a'])**2 + (row['c3_b']-row['c5_b'])**2)
        
        feat = {
            'contrast_bg_primary': abs(row['c1_L'] - row['c3_L']),
            'contrast_bg_accent': abs(row['c1_L'] - row['c5_L']),
            'de_bg_surf': de_bg_surf,
            'hue_dist_p_s': get_hue_distance(row['c3_h'], row['c4_h']),
            'hue_dist_p_a': get_hue_distance(row['c3_h'], row['c5_h']),
            'accent_vibrancy': row['c5_s'] * row['c5_v'],
            'bg_lightness': row['c1_L'],
            'de_primary_sec': de_p_s,
            'de_primary_acc': de_p_a
        }
        features.append(feat)
    return pd.DataFrame(features)

# -------------------------------------------------------------------
# 2. HÀM TRỰC QUAN HÓA (VISUALIZATION)
# -------------------------------------------------------------------
def visualize_modern_ui_model():
    data_file = os.path.join(RAW_DATA_PATH, "data_modern_ui.csv")
    model_file = os.path.join(MODEL_SAVE_PATH, "model_modern_ui.json")
    
    if not (os.path.exists(data_file) and os.path.exists(model_file)):
        print("❌ Thiếu file data hoặc model. Hãy đảm bảo đã chạy Gen Data và Train Model.")
        return

    print("[*] Đang nạp dữ liệu và mô hình Modern UI...")
    df_raw = pd.read_csv(data_file)
    X = engineer_ui_features(df_raw)
    y = df_raw['target_score']
    
    # Chia test set 20% để đánh giá khách quan
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Nạp "Bộ não"
    model = XGBRegressor()
    model.load_model(model_file)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # --- BIỂU ĐỒ 1: ACTUAL VS PREDICTED (Độ chính xác) ---
    print("[*] Đang vẽ biểu đồ Hồi quy (Regression Plot)...")
    plt.figure(figsize=(10, 8))
    # Chỉnh màu sắc sang tone "Modern" (Xanh Teal)
    sns.regplot(x=y_test, y=y_pred, 
                scatter_kws={'alpha':0.5, 'color':'#00b4d8', 's': 30}, 
                line_kws={'color':'#ff4d6d', 'linewidth': 2})
    
    plt.title(f"AI Performance Validation - Modern UI Evaluator\n$R^2$ Score: {r2:.4f}", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Điểm UI/UX thực tế (Ground Truth)", fontsize=12)
    plt.ylabel("AI Dự đoán (Predicted)", fontsize=12)
    
    # Kẻ lưới mờ để trông kỹ thuật hơn
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZE_PATH, "eval_modern_ui_regression.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- BIỂU ĐỒ 2: FEATURE IMPORTANCE (Tư duy của AI) ---
    print("[*] Đang vẽ biểu đồ Đặc trưng (Feature Importance)...")
    plt.figure(figsize=(12, 7))
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    
    # Đổi tên các nhãn cho chuyên nghiệp và dễ hiểu khi đưa vào báo cáo
    feature_names_mapping = {
        'contrast_bg_primary': 'Contrast: Background vs Primary',
        'contrast_bg_accent': 'Contrast: Background vs Accent',
        'hue_dist_p_a': 'Hue Angle: Primary vs Accent',
        'de_bg_surf': 'Separation: Background vs Surface',
        'hue_dist_p_s': 'Hue Angle: Primary vs Secondary',
        'bg_lightness': 'Theme: Background Lightness',
        'accent_vibrancy': 'Pop: Accent Vibrancy',
        'de_primary_acc': 'Diversity: Primary vs Accent',
        'de_primary_sec': 'Diversity: Primary vs Secondary'
    }
    importances.index = [feature_names_mapping.get(idx, idx) for idx in importances.index]
    
    # Vẽ biểu đồ thanh ngang
    ax = importances.plot(kind='barh', color='#0077b6', edgecolor='none')
    plt.title("AI Logic Decoding: Top Feature Importances", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Mức độ ảnh hưởng (Relative Importance)", fontsize=12)
    plt.ylabel("Đặc trưng UI/UX", fontsize=12)
    
    # Thêm giá trị cụ thể ở đuôi mỗi thanh
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", 
                    (p.get_width() + 0.005, p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, color='#333333')
        
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZE_PATH, "eval_modern_ui_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[+] XONG! Đã xuất 2 biểu đồ sắc nét (300 DPI) vào thư mục '{VISUALIZE_PATH}'.")

if __name__ == "__main__":
    visualize_modern_ui_model()