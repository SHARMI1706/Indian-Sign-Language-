# 🚀 ISL Translator - Complete Deployment Instructions

## Quick Start Commands

### Windows:
```bash
# 1. Navigate to project folder
cd path\to\isl-translator

# 2. Run the start script
start.bat
```

### macOS/Linux:
```bash
# 1. Navigate to project folder
cd path/to/isl-translator

# 2. Make script executable
chmod +x start.sh

# 3. Run the start script
./start.sh
```

### Manual Setup (All Platforms):
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

---

## 📂 Complete File Structure

After setup, your project should look like this:

```
isl-translator/
│
├── app.py                      # Flask backend server
├── requirements.txt            # Python dependencies
├── isl_digit_svm_model.pkl    # YOUR TRAINED MODEL (Required!)
├── svm_fixed.py               # Standalone test script
├── README.md                  # Documentation
├── start.sh                   # Linux/Mac start script
├── start.bat                  # Windows start script
│
├── templates/
│   └── index.html             # Frontend interface (WIDER UI)
│
└── venv/                      # Virtual environment (created automatically)
```

---

## 🔧 Detailed Setup Steps

### Step 1: Prepare Your Files

1. **Create project folder:**
   ```bash
   mkdir isl-translator
   cd isl-translator
   ```

2. **Copy all files to this folder:**
   - app.py
   - requirements.txt
   - svm_fixed.py
   - README.md
   - start.sh (Linux/Mac)
   - start.bat (Windows)
   - **isl_digit_svm_model.pkl** ⚠️ IMPORTANT!

3. **Create templates folder:**
   ```bash
   mkdir templates
   ```

4. **Place index.html in templates folder:**
   ```
   templates/index.html
   ```

### Step 2: Install Python Dependencies

#### Option A: Using Start Script (Easiest)

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

#### Option B: Manual Installation

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. **Install packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Verify Installation

Check if all packages are installed:
```bash
pip list
```

You should see:
- Flask
- flask-cors
- opencv-python
- mediapipe
- numpy
- joblib
- scikit-learn

### Step 4: Run the Application

```bash
python app.py
```

Expected output:
```
✅ SVM model loaded successfully
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
```

### Step 5: Access the Application

Open your browser and go to:
```
http://localhost:5000
```

---

## 🎯 Using the Application

1. **Welcome Screen:**
   - Click "Get Started"

2. **Language Selection:**
   - Choose your preferred language (English/Tamil/Hindi)
   - Click "Continue"

3. **Main App - Camera Tab:**
   - Click "Start Camera"
   - Allow camera permissions when prompted
   - Position your hand showing ISL digits (0-9)
   - See real-time recognition results

4. **View Results:**
   - Detected digit displays in large text
   - Confidence score shown
   - Recognition history tracked

---

## 🌐 Network Access (Optional)

To access from other devices on your network:

### Step 1: Find Your IP Address

**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" (e.g., 192.168.1.100)

**macOS:**
```bash
ifconfig | grep inet
```

**Linux:**
```bash
ip addr show
```

### Step 2: Update Frontend API URL

Open `templates/index.html` and find this line (around line 265):
```javascript
const API_URL = 'http://localhost:5000';
```

Change to:
```javascript
const API_URL = 'http://YOUR_IP:5000';
```

Example:
```javascript
const API_URL = 'http://192.168.1.100:5000';
```

### Step 3: Access from Other Devices

From any device on the same network, open:
```
http://YOUR_IP:5000
```

⚠️ **Note:** HTTPS is required for camera access on non-localhost domains. For local network testing, this limitation may vary by browser.

---

## 🎨 UI Customization

### Making UI Even Wider

The current UI uses `max-w-7xl` (1280px max width). To make it wider:

**Option 1: Full width**
In `templates/index.html`, line ~147:
```html
<!-- Change from: -->
<div class="h-full flex flex-col max-w-7xl mx-auto">

<!-- To: -->
<div class="h-full flex flex-col max-w-full mx-auto px-8">
```

**Option 2: Custom width**
```html
<div class="h-full flex flex-col mx-auto px-8" style="max-width: 1600px;">
```

**Option 3: Ultra-wide**
```html
<div class="h-full flex flex-col max-w-screen-2xl mx-auto px-8">
```

### Adjusting Camera Size

In `templates/index.html`, find the camera container (around line 199):
```html
<div class="relative rounded-2xl overflow-hidden webcam-container aspect-video mb-6">
```

Change `aspect-video` to:
- `aspect-square` - Square camera view
- `aspect-[4/3]` - 4:3 ratio
- `aspect-[21/9]` - Ultra-wide

### Changing Colors

The app uses a purple/indigo gradient theme. To change:

1. **Primary gradient:**
   ```css
   .gradient-bg {
     background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #6B8DD6 100%);
   }
   ```

2. **Button colors:**
   Find `from-indigo-500 to-purple-500` and replace with your colors

3. **Use Tailwind color classes:**
   - `bg-blue-500`, `bg-green-500`, `bg-red-500`, etc.

---

## 🔍 Testing the Model

### Test 1: Standalone Script

Before running the web app, test your model:
```bash
python svm_fixed.py
```

- Should open webcam window
- Shows "Digit: X" when hand is detected
- Press 'Q' to quit

### Test 2: Backend API

1. Start the server:
   ```bash
   python app.py
   ```

2. Test health endpoint:
   ```bash
   curl http://localhost:5000/api/health
   ```
   
   Expected response:
   ```json
   {"model_loaded":true,"status":"ok"}
   ```

### Test 3: Full Integration

1. Open browser: `http://localhost:5000`
2. Go through welcome → language → camera
3. Start camera and test digits 0-9
4. Verify recognition works

---

## 🐛 Common Issues & Solutions

### Issue 1: "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated
# You should see (venv) in your terminal

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue 2: "Model file not found"

**Solution:**
```bash
# Check if model exists in project root
ls -la isl_digit_svm_model.pkl  # Linux/Mac
dir isl_digit_svm_model.pkl     # Windows

# Verify path in app.py is correct
# Model should be in same directory as app.py
```

### Issue 3: Camera not working in browser

**Solution:**
1. Use Chrome or Edge (best support)
2. Check browser permissions: Settings → Privacy → Camera
3. Make sure no other app is using the camera
4. Try refreshing the page
5. Clear browser cache

### Issue 4: "Connection refused" error

**Solution:**
```bash
# 1. Verify backend is running
# Look for "Running on http://..." message

# 2. Check if port 5000 is already in use
# Windows:
netstat -ano | findstr :5000

# Linux/Mac:
lsof -i :5000

# 3. Kill the process or use different port
# In app.py, change:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue 5: Poor recognition accuracy

**Solutions:**
1. **Lighting:** Ensure bright, even lighting
2. **Background:** Use plain, contrasting background
3. **Distance:** Keep hand 1-2 feet from camera
4. **Clarity:** Make clear, distinct gestures
5. **Stability:** Hold gesture steady for 1-2 seconds
6. **Calibration:** May need to retrain model with more data

### Issue 6: Slow performance

**Solutions:**
```javascript
// In templates/index.html, increase recognition interval
// Change from 500ms to 1000ms or 1500ms

recognitionInterval = setInterval(async () => {
  await recognizeSign();
}, 1000); // Increased from 500
```

---

## 📊 Performance Optimization

### For Faster Recognition:

1. **Reduce camera resolution:**
   ```javascript
   // In templates/index.html
   video: { 
     width: { ideal: 640 },   // Reduced from 1280
     height: { ideal: 480 }   // Reduced from 720
   }
   ```

2. **Adjust image quality:**
   ```javascript
   // In recognizeSign() function
   const imageData = canvas.toDataURL('image/jpeg', 0.6); // Reduced from 0.8
   ```

3. **Increase recognition interval:**
   ```javascript
   recognitionInterval = setInterval(async () => {
     await recognizeSign();
   }, 1000); // Increased interval
   ```

---

## 🔒 Security for Production

### For production deployment:

1. **Disable debug mode:**
   ```python
   # In app.py
   app.run(debug=False, host='0.0.0.0', port=5000)
   ```

2. **Use production server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Enable HTTPS:**
   - Required for camera access on public domains
   - Use nginx with SSL certificate
   - Or use cloud platforms with built-in HTTPS

4. **Add rate limiting:**
   ```bash
   pip install Flask-Limiter
   ```

5. **Environment variables:**
   ```python
   # Use environment variables for sensitive data
   import os
   app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
   ```

---

## 📱 Mobile Deployment

### Local Network (Mobile Browser):

1. Connect mobile to same WiFi as computer
2. Find computer's IP address
3. Update API_URL in index.html
4. Access from mobile: `http://COMPUTER_IP:5000`

⚠️ **Note:** Mobile camera support varies by browser. Chrome Mobile works best.

---

## 🚀 Cloud Deployment

### Heroku Deployment:

1. **Create Procfile:**
   ```
   web: gunicorn app:app
   ```

2. **Create runtime.txt:**
   ```
   python-3.10.12
   ```

3. **Update requirements.txt:**
   Add `gunicorn` to requirements

4. **Deploy:**
   ```bash
   heroku create your-app-name
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

5. **Configure buildpacks:**
   ```bash
   heroku buildpacks:set heroku/python
   ```

### Railway/Render Deployment:

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn app:app`
4. Deploy automatically on git push

---

## 📝 Additional Notes

### ISL Digit Guidelines:

| Digit | Hand Position |
|-------|---------------|
| 0 | Closed fist |
| 1 | Index finger extended |
| 2 | Index + Middle fingers |
| 3 | Index + Middle + Ring |
| 4 | All fingers except thumb |
| 5 | All fingers extended |
| 6-9 | Follow ISL standard gestures |

### Browser Recommendations:

✅ **Best:** Chrome, Edge (Chromium)
✅ **Good:** Firefox
⚠️ **Limited:** Safari
❌ **Not recommended:** IE, older browsers

---

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section
2. Verify all files are in correct locations
3. Check browser console (F12) for errors
4. Verify backend logs in terminal
5. Test model with standalone script first
6. Ensure camera permissions are granted

---

## ✅ Deployment Checklist

- [ ] Python 3.8+ installed
- [ ] All files copied to project folder
- [ ] Model file (isl_digit_svm_model.pkl) present
- [ ] templates/ folder created
- [ ] index.html in templates/ folder
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Backend starts without errors
- [ ] Can access http://localhost:5000
- [ ] Camera permissions granted
- [ ] Model recognition working
- [ ] UI displays correctly

---

**Happy Deploying! 🎉**

For questions or issues, refer to the README.md file or troubleshooting sections above.