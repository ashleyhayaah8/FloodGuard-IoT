import telebot
import pandas as pd
import joblib
import os
import time
import threading
import warnings
from datetime import datetime, timedelta
from telebot import apihelper

# ==========================================
# 1. SETUP & ROBUST PROXY
# ==========================================
apihelper.proxy = {'https': 'http://proxy.server:3128'}
apihelper.CONNECT_TIMEOUT = 40 
apihelper.READ_TIMEOUT = 40

warnings.filterwarnings('ignore')

API_TOKEN = '8963411563:AAHmJgjALQumVmEVBIdB5S1QpGr2qWzpuH4'
bot = telebot.TeleBot(API_TOKEN)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'floodguard_train_ready_full.csv')

# Load Models
models = {}
try:
    models = {
        '1h': joblib.load(os.path.join(BASE_DIR, 'model_1h.pkl')),
        '2h': joblib.load(os.path.join(BASE_DIR, 'model_2h.pkl')),
        '3h': joblib.load(os.path.join(BASE_DIR, 'model_3h.pkl'))
    }
    print("[OK] Model ML berhasil dimuat")
except Exception as e:
    print(f"[ERROR MODEL] {e}")

TW, TB = 171.3, 222.7
simulasi_aktif = False
waktu_simulasi_skrg = None
user_target_id = None

# ==========================================
# 2. LOGIKA PREDIKSI & STATUS
# ==========================================

def get_status(wl):
    if wl >= TB: return "BAHAYA", "🔴"
    if wl >= TW: return "WASPADA", "🟡"
    return "AMAN", "🟢"

def predict_logic(target_datetime_str):
    try:
        df = pd.read_csv(CSV_FILE)
        rename_map = {
            'rainfall_1h_mm': 'rainfall_1h_max', 'rainfall_3h_mm': 'rainfall_3h_max',
            'rainfall_6h_mm': 'rainfall_6h_max', 'trend_hilir_cm_h': 'trend_hilir_cm_d',
            'trend_hulu_cm_h': 'trend_hulu_cm_d', 'wl_hilir_lag1h': 'wl_hilir_lag1d',
            'wl_hilir_lag2h': 'wl_hilir_lag2d', 'wl_hilir_lag3h': 'wl_hilir_lag3d',
            'wl_hulu_lag1h': 'wl_hulu_lag1d', 'wl_hulu_lag2h': 'wl_hulu_lag2d',
            'wl_hulu_lag3h': 'wl_hulu_lag3d', 'rain_lag1h': 'rain_lag1d',
            'rain_lag2h': 'rain_lag2d', 'rain_lag3h': 'rain_lag3d',
            'wl_hilir_ma3h': 'wl_hilir_ma3', 'wl_hulu_ma3h': 'wl_hulu_ma3'
        }
        df = df.rename(columns=rename_map)
        match = df[df['datetime'] == target_datetime_str]
        if match.empty: return None
        
        row = match.iloc[0]
        feats = list(models['1h'].feature_names_in_)
        X = pd.DataFrame([row[feats]], columns=feats)
        w_min, w_max = df['water_level_hilir_cm_raw'].min(), df['water_level_hilir_cm_raw'].max()
        
        p1 = w_min + models['1h'].predict(X)[0] * (w_max - w_min)
        p2 = w_min + models['2h'].predict(X)[0] * (w_max - w_min)
        p3 = w_min + models['3h'].predict(X)[0] * (w_max - w_min)
        
        return {'wl_now': round(row['water_level_hilir_cm_raw'], 2), 'p1': round(p1, 2), 'p2': round(p2, 2), 'p3': round(p3, 2)}
    except Exception: return None

# ==========================================
# 3. FUNGSI KIRIM PESAN (DENGAN PROTEKSI)
# ==========================================

def safe_send(u_id, text):
    """Mengirim pesan dengan proteksi jika proxy sedang 503"""
    try:
        bot.send_message(u_id, text, parse_mode='Markdown')
    except Exception as e:
        print(f"[RETRY] Proxy sibuk, gagal kirim pesan: {e}")

def send_full_report(u_id, t_obj, data):
    resp = f"📊 *LAPORAN FLOODGUARD*\n🕒 Waktu: `{t_obj.strftime('%Y-%m-%d %H:%M')}`\n------------------------\n\n"
    preds = [(1, data['p1']), (2, data['p2']), (3, data['p3'])]
    for jam, nilai in preds:
        w_nanti = (t_obj + timedelta(hours=jam)).strftime("%H:%M")
        st, em = get_status(nilai)
        resp += f"📍 Pukul {w_nanti} WIB\n└ Prediksi: {nilai} cm\n└ Status: {em} {st}\n\n"
    resp += "_Sistem Terintegrasi Cloud & Data-Driven IoT_"
    safe_send(u_id, resp)

def send_alert_only(u_id, t_obj, data):
    stat_p3, em_p3 = get_status(data['p3'])
    if stat_p3 != "AMAN":
        w_banjir = (t_obj + timedelta(hours=3)).strftime('%H:%M')
        pesan = (f"🔔 *NOTIFIKASI SIMULASI* ({t_obj.strftime('%H:%M')})\n"
                 f"Terdeteksi potensi {em_p3} *{stat_p3}* pada pukul *{w_banjir}*!\n"
                 f"Prediksi: `{data['p3']} cm`. Segera bersiap dalam 1 jam! ⏳")
        safe_send(u_id, pesan)

# ==========================================
# 4. THREAD SIMULASI & HANDLERS
# ==========================================

def simulation_thread():
    global simulasi_aktif, waktu_simulasi_skrg, user_target_id
    while True:
        if simulasi_aktif and waktu_simulasi_skrg:
            t_str = waktu_simulasi_skrg.strftime('%Y-%m-%d %H:%M')
            data = predict_logic(t_str)
            if data:
                send_alert_only(user_target_id, waktu_simulasi_skrg, data)
                waktu_simulasi_skrg += timedelta(hours=1)
            else:
                simulasi_aktif = False
        time.sleep(10)

@bot.message_handler(commands=['cek'])
def handle_cek(message):
    global simulasi_aktif, waktu_simulasi_skrg, user_target_id
    text = message.text.replace('/cek', '').strip()
    if not text:
        bot.reply_to(message, "Gunakan format: `/cek YYYY-MM-DD HH:mm`")
        return
    try:
        waktu_simulasi_skrg = datetime.strptime(text, '%Y-%m-%d %H:%M')
        user_target_id = message.chat.id
        data = predict_logic(text)
        if data:
            send_full_report(user_target_id, waktu_simulasi_skrg, data)
            simulasi_aktif = True
            safe_send(user_target_id, "🚀 *Simulasi Background Aktif.*")
        else:
            bot.reply_to(message, "❌ Data tidak ditemukan.")
    except Exception as e:
        bot.reply_to(message, f"Format salah: {e}")

# ==========================================
# 5. RUN
# ==========================================
if __name__ == "__main__":
    threading.Thread(target=simulation_thread, daemon=True).start()
    print("--- Bot FloodGuard Siap ---")
    while True:
        try:
            bot.polling(none_stop=True, interval=2, timeout=40)
        except Exception as e:
            print(f"[POLLING ERROR] {e}")
            time.sleep(20) # Jeda lebih lama jika proxy 503