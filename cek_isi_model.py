import joblib
import os
from pprint import pprint

# Daftar model yang ingin dicek
models_to_check = ['model_1h.pkl', 'model_2h.pkl', 'model_3h.pkl']

print("="*60)
print("🔍 FLOODGUARD: MODEL INSPECTOR")
print("="*60)

for model_name in models_to_check:
    if not os.path.exists(model_name):
        print(f"\n⚠️  File {model_name} tidak ditemukan di folder ini.")
        continue

    print(f"\n>>> MEMBEDAH MODEL: {model_name} <<<")
    
    try:
        # Load Model
        model = joblib.load(model_name)
        
        # 1. Tipe Algoritma
        print(f"\n[1] Algoritma:")
        print(f"    {type(model).__name__}")

        # 2. Fitur/Kolom yang Dipelajari
        print(f"\n[2] Fitur yang Dibutuhkan (Input):")
        if hasattr(model, 'feature_names_in_'):
            feats = list(model.feature_names_in_)
            for i, f in enumerate(feats, 1):
                print(f"    {i}. {f}")
        else:
            print("    Model tidak menyimpan metadata nama fitur.")

        # 3. Parameter Utama (Konfigurasi)
        print(f"\n[3] Parameter Konfigurasi (Internal):")
        params = model.get_params()
        # Kita ambil 3 parameter paling penting saja agar tidak kepanjangan
        important_params = ['n_estimators', 'max_depth', 'random_state']
        for p in important_params:
            if p in params:
                print(f"    - {p}: {params[p]}")

        # 4. Feature Importance (Variabel Paling Berpengaruh)
        if hasattr(model, 'feature_importances_'):
            print(f"\n[4] 3 Variabel Paling Berpengaruh (Top 3 Priority):")
            importances = sorted(
                zip(model.feature_names_in_, model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )
            for i, (name, val) in enumerate(importances[:3], 1):
                print(f"    {i}. {name} ({val*100:.2f}%)")

    except Exception as e:
        print(f"❌ Gagal membedah {model_name}: {e}")

    print("\n" + "-"*40)

print("\n" + "="*60)
print("INSPEKSI SELESAI")
print("="*60)