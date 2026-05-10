import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Thiết lập đường dẫn
RAW_DATA_PATH = os.path.join("data", "raw")
MODEL_SAVE_PATH = os.path.join("models", "saved_models")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def get_hue_distance(h1, h2):
    diff = abs(h1 - h2) % 360
    return diff if diff <= 180 else 360 - diff

def engineer_ui_features(df):
    """Trích xuất các đặc trưng chuẩn UI/UX từ dữ liệu 5 màu"""
    features = []
    for _, row in df.iterrows():
        # Định danh vai trò:
        # 1: Background, 2: Surface, 3: Primary, 4: Secondary, 5: Accent
        
        # 1. Khoảng cách thị giác Delta E (Phát hiện màu trùng lặp/Dead palette)
        de_bg_surf = np.sqrt((row['c1_L']-row['c2_L'])**2 + (row['c1_a']-row['c2_a'])**2 + (row['c1_b']-row['c2_b'])**2)
        de_p_s = np.sqrt((row['c3_L']-row['c4_L'])**2 + (row['c3_a']-row['c4_a'])**2 + (row['c3_b']-row['c4_b'])**2)
        de_p_a = np.sqrt((row['c3_L']-row['c5_L'])**2 + (row['c3_a']-row['c5_a'])**2 + (row['c3_b']-row['c5_b'])**2)
        
        feat = {
            # Tiêu chí Accessibility (Độ tương phản chói giữa Nền và Chữ/Nút)
            'contrast_bg_primary': abs(row['c1_L'] - row['c3_L']),
            'contrast_bg_accent': abs(row['c1_L'] - row['c5_L']),
            
            # Tiêu chí Layering (Sự phân tách giữa Nền và Card)
            'de_bg_surf': de_bg_surf,
            
            # Tiêu chí Nhận diện thương hiệu (Tương đồng giữa Primary và Secondary)
            'hue_dist_p_s': get_hue_distance(row['c3_h'], row['c4_h']),
            
            # Tiêu chí Call-to-action (Sự nổi bật của Accent)
            'hue_dist_p_a': get_hue_distance(row['c3_h'], row['c5_h']),
            'accent_vibrancy': row['c5_s'] * row['c5_v'], # Accent phải rực rỡ
            
            # Tính chất Theme (Light mode hay Dark mode)
            'bg_lightness': row['c1_L'],
            
            # Đa dạng màu sắc (Tránh dead palette)
            'de_primary_sec': de_p_s,
            'de_primary_acc': de_p_a
        }
        features.append(feat)
    return pd.DataFrame(features)

def train_modern_ui_model():
    file_path = os.path.join(RAW_DATA_PATH, "data_modern_ui.csv")
    if not os.path.exists(file_path): 
        print("❌ Không tìm thấy file data_modern_ui.csv. Hãy chạy data_synthesizer.py trước!")
        return
    
    print("\n[*] Đang nạp dữ liệu Modern UI (5000 mẫu)...")
    df_raw = pd.read_csv(file_path)
    X = engineer_ui_features(df_raw)
    y = df_raw['target_score']
    
    print("[*] Đang đánh giá chéo mô hình (5-Fold CV)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    r2_scores, mae_scores, rmse_scores = [], [], []
    best_model, best_r2 = None, -float('inf')
    
    # Cấu hình XGBoost tối ưu cho dữ liệu có nhiễu thực tế
    base_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,         # Tăng độ sâu vì logic UI phức tạp hơn
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=2.0,      # Phạt L2 mạnh hơn để chống Overfitting
        random_state=42,
        n_jobs=-1
    )

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        base_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = base_model.predict(X_test)
        
        current_r2 = r2_score(y_test, y_pred)
        r2_scores.append(current_r2)
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_model = base_model

    # Báo cáo kết quả
    print("\n--- KẾT QUẢ HUẤN LUYỆN (MODERN UI MODEL) ---")
    print(f"    - R-Squared (CV Mean): {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"    - MAE (CV Mean):       {np.mean(mae_scores):.4f}")
    print(f"    - RMSE (CV Mean):      {np.mean(rmse_scores):.4f}")
    
    # Đặc trưng quan trọng nhất
    importances = best_model.feature_importances_
    top_features = X.columns[np.argsort(importances)[-3:]][::-1] # Lấy top 3
    print(f"    - Top 3 Đặc trưng quyết định UI/UX:")
    for i, feat in enumerate(top_features):
        print(f"      {i+1}. {feat}")
        
    # Lưu mô hình
    save_path = os.path.join(MODEL_SAVE_PATH, "model_modern_ui.json")
    best_model.save_model(save_path)
    print(f"\n[+] Đã lưu 'Bộ não UI/UX' tại: {save_path}")

if __name__ == "__main__":
    train_modern_ui_model()