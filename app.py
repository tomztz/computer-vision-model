from flask import Flask, request
import io
app = Flask(__name__)
from datetime import datetime
import detect_video

@app.route('/', methods=['PUT'])
def index():
    raw_data = request.get_data()
    data = raw_data.decode()
    now = datetime.now()
    filename = "video-" + now.strftime("%y-%m-%d %H:%M:%S") + ".mp4"
    new_data = [int(num) for num in data[1:-1].split(",")]
    with open("./" + filename, "wb") as file:
        file.write(bytearray(new_data))
    result = detect_video.main("./" + filename)
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0")
    