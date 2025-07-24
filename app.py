import os
import json
import time
from flask import Flask, render_template, request, Response, jsonify
from async_forum_scraper import AsyncForumScraper
from text_model import TextModel  # Import our custom model instead
import asyncio
import threading

app = Flask(__name__)

# Global state for tracking progress
scraping_progress = {
    "is_running": False,
    "total_posts": 0,
    "processed_posts": 0,
    "successful_posts": 0,
    "failed_posts": 0,
    "current_id": 0,
    "start_time": 0,
    "events": [],
    "connection_stats": {
        "success": 0,
        "failure": 0,
        "avg_response_time": 0,
        "total_response_time": 0
    }
}

training_progress = {
    "is_running": False,
    "total_steps": 0,
    "current_step": 0,
    "loss": 0.0,
    "start_time": 0,
    "status": ""
}

# Event listeners for SSE
listeners = []

# Function to send SSE events to all connected clients
def send_event(event_type, data):
    if not listeners:
        return
    
    event_data = json.dumps(data)
    message = f"event: {event_type}\ndata: {event_data}\n\n"
    
    # Create a copy to avoid modification during iteration
    for listener in list(listeners):
        try:
            # Simply append the message to the list
            listener.append(message)
        except Exception:
            # Remove the listener if there's an error
            if listener in listeners:
                listeners.remove(listener)

# Progress callback for scraper
def scraping_progress_callback(progress_type, data):
    global scraping_progress
    
    if progress_type == "login":
        # Handle login progress events
        event = {
            "type": "login",
            "success": data.get("success", False),
            "username": data.get("username", "Unknown"),
            "message": "Login successful" if data.get("success", False) else data.get("error", "Login failed")
        }
        scraping_progress["events"].insert(0, event)
        scraping_progress["events"] = scraping_progress["events"][:50]
        
    elif progress_type == "start":
        scraping_progress["is_running"] = True
        scraping_progress["total_posts"] = data["total_posts"]
        scraping_progress["start_time"] = time.time()
        scraping_progress["events"] = []
        scraping_progress["connection_stats"] = {
            "success": 0,
            "failure": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
    
    elif progress_type == "post_processed":
        scraping_progress["processed_posts"] += 1
        scraping_progress["current_id"] = data["post_id"]
        
        if data["success"]:
            scraping_progress["successful_posts"] += 1
            event = {
                "type": "success",
                "post_id": data["post_id"],
                "thread_id": data.get("thread_id", "N/A"),
                "username": data.get("username", "N/A"),
                "title": data.get("title", "N/A")[:30] + "..." if data.get("title") and len(data.get("title")) > 30 else data.get("title", "N/A"),
                "content_length": len(data.get("content", ""))
            }
        else:
            scraping_progress["failed_posts"] += 1
            event = {
                "type": "failure",
                "post_id": data["post_id"],
                "error": data.get("error", "Unknown error")
            }
        
        scraping_progress["events"].insert(0, event)
        # Keep only the last 50 events
        scraping_progress["events"] = scraping_progress["events"][:50]
    
    elif progress_type == "connection":
        if data["success"]:
            scraping_progress["connection_stats"]["success"] += 1
        else:
            scraping_progress["connection_stats"]["failure"] += 1
        
        if "response_time" in data:
            scraping_progress["connection_stats"]["total_response_time"] += data["response_time"]
            total = scraping_progress["connection_stats"]["success"] + scraping_progress["connection_stats"]["failure"]
            if total > 0:
                scraping_progress["connection_stats"]["avg_response_time"] = scraping_progress["connection_stats"]["total_response_time"] / total
    
    elif progress_type == "complete":
        scraping_progress["is_running"] = False
    
    # Send the updated progress to all clients
    send_event("scraping_progress", scraping_progress)

# Progress callback for training
def training_progress_callback(progress_type, data):
    global training_progress
    
    if progress_type == "start":
        training_progress["is_running"] = True
        training_progress["total_steps"] = data["total_steps"]
        training_progress["current_step"] = 0
        training_progress["loss"] = 0.0
        training_progress["start_time"] = time.time()
        training_progress["status"] = "Initializing model..."
    
    elif progress_type == "step":
        training_progress["current_step"] = data["step"]
        training_progress["loss"] = data["loss"]
        training_progress["status"] = f"Training step {data['step']}/{training_progress['total_steps']}"
    
    elif progress_type == "complete":
        training_progress["is_running"] = False
        training_progress["status"] = "Training complete!"
    
    # Send the updated progress to all clients
    send_event("training_progress", training_progress)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/events')
def events():
    def event_stream():
        # Use a simple list instead of asyncio.Queue
        queue = []
        queue_id = id(queue)
        listeners.append(queue)
        
        try:
            # Send initial state
            yield f"event: scraping_progress\ndata: {json.dumps(scraping_progress)}\n\n"
            yield f"event: training_progress\ndata: {json.dumps(training_progress)}\n\n"
            
            # Keep the connection alive with a heartbeat
            last_heartbeat = time.time()
            while True:
                # Check if there are messages in the queue
                if queue:
                    message = queue.pop(0)
                    yield message
                else:
                    # Send a heartbeat every 10 seconds to keep the connection alive
                    current_time = time.time()
                    if current_time - last_heartbeat >= 10:
                        yield f":heartbeat\n\n"
                        last_heartbeat = current_time
                    time.sleep(0.1)  # Small sleep to prevent CPU hogging
        except GeneratorExit:
            # Remove the queue from listeners when the connection is closed
            for i, listener in enumerate(listeners):
                if id(listener) == queue_id:
                    listeners.pop(i)
                    break
    
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/scrape', methods=['POST'])
def scrape():
    if scraping_progress["is_running"]:
        return jsonify({"error": "Scraping is already in progress"}), 400
    
    try:
        start_id = int(request.form.get('start_id', 1))
        end_id = int(request.form.get('end_id', 100))
        max_connections = int(request.form.get('max_connections', 5000))
        
        # Optional authentication parameters
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        use_auth = bool(username and password)
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400
    
    # Validate parameters
    if start_id < 1 or end_id < start_id or max_connections < 1 or max_connections > 5000:
        return jsonify({"error": "Invalid parameters"}), 400
    
    # Start scraping in a background thread
    def run_scraper():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Use FixedAsyncForumScraper with authentication if credentials provided
        if use_auth:
            from fixed_async_forum_scraper import FixedAsyncForumScraper
            scraper = FixedAsyncForumScraper(
                progress_callback=scraping_progress_callback,
                username=username,
                password=password
            )
        else:
            scraper = AsyncForumScraper(progress_callback=scraping_progress_callback)
        
        results = loop.run_until_complete(scraper.scrape_range(start_id, end_id, max_connections))
        
        # Save results
        scraper.save_data(results['posts'], results['failed_ids'], 'forum_data.json')
        
        loop.close()
    
    thread = threading.Thread(target=run_scraper)
    thread.daemon = True
    thread.start()
    
    auth_status = "with authentication" if use_auth else "without authentication"
    return jsonify({"status": f"Scraping started {auth_status}"})

@app.route('/train', methods=['POST'])
def train():
    if training_progress["is_running"]:
        return jsonify({"error": "Training is already in progress"}), 400
    
    if not os.path.exists('forum_data.json'):
        return jsonify({"error": "No data available. Please scrape data first."}), 400
    
    try:
        epochs = int(request.form.get('epochs', 5))
        batch_size = int(request.form.get('batch_size', 16))
        learning_rate = float(request.form.get('learning_rate', 0.001))
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400
    
    # Validate parameters
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        return jsonify({"error": "Invalid parameters"}), 400
    
    # Start training in a background thread
    def run_training():
        model = TextModel(progress_callback=training_progress_callback)
        model.train('forum_data.json', epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        model.save_model('forum_model.pkl')
    
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "Training started"})

@app.route('/generate', methods=['POST'])
def generate():
    if not os.path.exists('forum_model.pkl'):
        return jsonify({"error": "No model available. Please train the model first."}), 400
    
    prompt = request.form.get('prompt', '')
    max_length = int(request.form.get('max_length', 1000))
    temperature = float(request.form.get('temperature', 0.7))
    
    # Validate parameters
    if max_length < 1 or max_length > 1000 or temperature <= 0 or temperature > 2:
        return jsonify({"error": "Invalid parameters"}), 400
    
    # Load model and generate text
    model = TextModel()
    model.load_model('forum_model.pkl')
    generated_text = model.generate(prompt, max_length, temperature)
    
    return jsonify({"text": generated_text})

if __name__ == '__main__':
    app.run(debug=True)