import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf

# Initialize Flask application
app = Flask(__name__)

# Configure uploads directory and allowed files
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variables
model = None
class_names = []
butterfly_metadata = {
    "ADONIS": {
        "scientific_name": "Polyommatus bellargus",
        "common_name": "Kupu-kupu Biru Adonis",
        "description": "Kupu-kupu Biru Adonis adalah spesies kupu-kupu kecil yang menakjubkan dalam famili Lycaenidae. Kupu-kupu jantan memiliki warna sayap biru langit yang cemerlang dan berkilau, sedangkan betinanya berwarna cokelat cokelat dengan sisik biru di dekat pangkal sayap dan bintik-bintik oranye di bagian tepi sayap.",
        "habitat": "Padang rumput kapur yang cerah dan hangat, bukit kapur, dan lereng curam yang menghadap ke selatan tempat tanaman inang ulatnya (Horseshoe Vetch) tumbuh subur.",
        "fun_fact": "Ulat kupu-kupu Biru Adonis mengeluarkan zat manis lengket yang disebut embun madu (honeydew). Semut secara aktif merawat dan melindungi ulat-ulat ini dari predator sebagai imbalan atas cairan manis tersebut."
    },
    "MESTRA": {
        "scientific_name": "Mestra hersilia",
        "common_name": "Kupu-kupu Pale Mestra",
        "description": "Mestra hersilia adalah kupu-kupu tropis yang ditandai dengan sayap tipis berwarna abu-abu pucat atau cokelat muda dengan pinggiran oranye-cokelat yang halus. Spesies ini menunjukkan pola terbang yang lambat dan melayang dekat dengan permukaan tanah.",
        "habitat": "Habitat terbuka, semak-semak padang rumput, pinggiran hutan kering tropis, dan kawasan pertanian yang terganggu.",
        "fun_fact": "Alih-alih mencari nektar bunga sebagai makanan utama, kupu-kupu Mestra sering terlihat mengumpulkan nutrisi dari buah-buahan busuk, getah pohon, tanah basah, atau kotoran burung."
    },
    "MONARCH": {
        "scientific_name": "Danaus plexippus",
        "common_name": "Kupu-kupu Raja (Monarch)",
        "description": "Sebagai salah satu spesies kupu-kupu paling terkenal di dunia, Monarch dapat langsung dikenali dari sayap oranye besarnya yang dihiasi urat hitam dan dikelilingi oleh baris ganda bintik-bintik putih yang indah.",
        "habitat": "Lapangan terbuka, padang rumput, taman kota, pinggir jalan, dan padang rumput liar tempat tanaman milkweed (satu-satunya sumber makanan ulatnya) tumbuh.",
        "fun_fact": "Kupu-kupu Monarch terkenal karena migrasi tahunan mereka yang epik. Jutaan Monarch melakukan perjalanan hingga 4.800 kilometer (3.000 mil) dari Amerika Utara untuk mencapai hutan tempat mereka menghabiskan musim dingin di Meksiko Tengah."
    },
    "PEACOCK": {
        "scientific_name": "Aglais io",
        "common_name": "Kupu-kupu Merak (Peacock)",
        "description": "Kupu-kupu Merak memiliki bintik menyerupai mata (eyespots) besar yang indah dan berkilau di permukaan atas sayap cokelat kemerahannya, meniru pola bulu ekor burung merak untuk mengelabui musuh.",
        "habitat": "Pinggiran hutan, padang rumput, padang gembala, taman, dan kebun. Ulatnya terutama memakan tanaman jelatang (stinging nettles).",
        "fun_fact": "Bintik menyerupai mata tersebut bertindak sebagai pertahanan visual yang intens. Saat terancam, kupu-kupu ini akan membuka sayapnya secara tiba-tiba dan memamerkan bintik mata besar tersebut untuk mengejutkan dan menakuti predator seperti burung dan tikus."
    },
    "ZEBRA_LONG_WING": {
        "scientific_name": "Heliconius charithonia",
        "common_name": "Kupu-kupu Sayap Panjang Zebra",
        "description": "Spesies ini memiliki sayap hitam panjang dan sempit yang dihiasi garis-garis kuning cerah yang mencolok. Mereka memiliki gaya terbang yang lambat, anggun, dan melayang tenang di udara.",
        "habitat": "Hutan tropis yang lembap (hammock), pinggiran hutan, jalan setapak yang teduh, dan kebun di daerah beriklim hangat.",
        "fun_fact": "Secara unik, Zebra Longwing mencerna serbuk sari selain menghisap nektar bunga. Pola makan serbuk sari yang kaya protein ini memungkinkan mereka hidup hingga enam bulan, jauh lebih lama dibandingkan kupu-kupu lain yang umumnya hanya bertahan hidup beberapa minggu."
    }
}

def load_ml_model():
    global model, class_names
    model_path = 'model/butterfly_model.keras'
    classes_path = 'model/classes.txt'
    
    # Check if model exists
    if os.path.exists(model_path):
        try:
            print("Loading trained Keras model...")
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print("Model file not found. Application will run in demo/fallback mode.")
        model = None
        
    # Load classes
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = ["ADONIS", "MESTRA", "MONARCH", "PEACOCK", "ZEBRA_LONG_WING"]
    print(f"Loaded class names: {class_names}")

# Load model upon startup
load_ml_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Load metrics from json if available
    metrics_path = 'model/evaluation_metrics.json'
    metrics_data = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error reading metrics JSON: {e}")
            
    return render_template('dashboard.html', metrics=metrics_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, class_names
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file yang diunggah"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Tidak ada file yang dipilih"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Format file tidak didukung. Silakan unggah gambar (PNG, JPG, JPEG, WEBP)."}), 400
        
    try:
        # Save file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # In case model isn't trained yet, provide a mock response for fallback testing
        if model is None:
            # Fallback mock prediction
            mock_class = np.random.choice(class_names)
            mock_prob = float(np.random.uniform(0.75, 0.99))
            mock_probs = {c: float(0.01 + (mock_prob-0.01 if c == mock_class else np.random.uniform(0, (1-mock_prob)/4))) for c in class_names}
            # Normalize mock_probs to sum to 1
            total = sum(mock_probs.values())
            mock_probs = {k: v/total for k, v in mock_probs.items()}
            
            metadata = butterfly_metadata.get(mock_class, {
                "scientific_name": "Tidak Diketahui",
                "common_name": mock_class,
                "description": "Model tidak dimuat. Menggunakan data simulasi fallback.",
                "habitat": "N/A",
                "fun_fact": "N/A"
            })
            
            return jsonify({
                "success": True,
                "fallback": True,
                "prediction": mock_class,
                "confidence": mock_prob,
                "image_url": '/' + file_path.replace('\\', '/'),
                "details": metadata,
                "probabilities": mock_probs
            })
            
        # Model is loaded, let's run inference!
        # Open image and preprocess
        img = Image.open(file_path).convert('RGB')
        img = img.resize((160, 160))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, 224, 224, 3)
        
        # Run inference
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        pred_class = class_names[pred_idx]
        confidence = float(preds[0][pred_idx])
        
        # Build full probability dictionary
        probabilities = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
        
        # Get butterfly details
        metadata = butterfly_metadata.get(pred_class, {
            "scientific_name": "Tidak Diketahui",
            "common_name": pred_class,
            "description": "Metadata tidak ditemukan.",
            "habitat": "Tidak Diketahui",
            "fun_fact": "Tidak Diketahui"
        })
        
        return jsonify({
            "success": True,
            "fallback": False,
            "prediction": pred_class,
            "confidence": confidence,
            "image_url": '/' + file_path.replace('\\', '/'),
            "details": metadata,
            "probabilities": probabilities
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Terjadi kesalahan saat memproses prediksi: {str(e)}"}), 500

if __name__ == '__main__':
    # Make sure model is loaded before running server
    if model is None:
        load_ml_model()
    app.run(debug=True, port=5000)
