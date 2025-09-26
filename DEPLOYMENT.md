# Deployment Guide

This guide explains how to deploy the MNIST Digit Recognition system to various platforms.

## Local Development

### Quick Start
```bash
# Clone the repository
git clone https://github.com/waqaszahoor1998/mnist.git
cd mnist

# Run setup script
python setup.py

# Start the web application
python web.py
```

### Manual Setup
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (optional - will train automatically if no model exists)
python improved_model.py

# 4. Run the application
python web.py
```

## Cloud Deployment

### Heroku

1. **Create a Procfile:**
```
web: python app.py
```

2. **Create runtime.txt:**
```
python-3.11.0
```

3. **Deploy:**
```bash
heroku create your-mnist-app
git push heroku main
heroku open
```

### Railway

1. **Create railway.json:**
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python app.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

2. **Deploy:**
```bash
railway login
railway init
railway up
```

### Docker Deployment

1. **Create Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
```

2. **Build and run:**
```bash
docker build -t mnist-app .
docker run -p 5001:5001 mnist-app
```

## Environment Variables

Set these environment variables for production:

- `FLASK_ENV=production`
- `PORT=5001` (for cloud platforms)
- `MODEL_PATH=./models/` (optional)

## Performance Optimization

### For Production:
1. **Use a production WSGI server:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

2. **Enable model caching:**
```python
# In app.py, add model caching
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model_cached():
    return load_model()
```

3. **Use a CDN for static files** (if deploying web interface)

## Monitoring

### Health Check Endpoint
Add to `app.py`:
```python
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})
```

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Troubleshooting

### Common Issues:

1. **Model not loading:**
   - Check if model files exist
   - Verify file permissions
   - Check disk space

2. **Memory issues:**
   - Reduce batch size in training
   - Use model quantization
   - Implement model caching

3. **Slow predictions:**
   - Use GPU acceleration
   - Implement prediction caching
   - Optimize model architecture

## Security Considerations

1. **Input validation:** Validate uploaded images
2. **Rate limiting:** Implement request rate limiting
3. **CORS:** Configure CORS properly for web interface
4. **Authentication:** Add authentication if needed

## Scaling

### Horizontal Scaling:
- Use load balancer
- Implement session management
- Use shared model storage

### Vertical Scaling:
- Increase server resources
- Use GPU acceleration
- Optimize model architecture
