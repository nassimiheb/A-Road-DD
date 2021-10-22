import sys
sys.path.append('./yolo')

from flask import Flask, request, jsonify
import threading

from core import trips_management

app = Flask(__name__, static_url_path='/static')
current_thread_id = 0

#Trigger to indecate control the detection script
exit_detect_event = threading.Event()
exit_detect_event.set()

# get all trips
TRIPS_FOLDER = './trips_data'
all_trips = trips_management.read_all_trips(TRIPS_FOLDER)
IMAGES_FOLDER = './trip_images_save'

#initialize position global variables
coor_lat = 0
coor_long = 0

#model
model_wights = 'yolov5s.pt'

def return_error(msg, code):
    return jsonify({
                    'error' : True, 
                    'message' : msg
                }),code

def detect_async(opt, file_save, img_save_path = None, save_img=True):
    pass

@app.route('/coordinates', methods=['PUT'])
def result():
    global coor_lat
    global coor_long
    request_json = request.get_json()
    lat_ = request_json.get('lat', None)
    long_ = request_json.get('long', None)

    print(lat_,long_)
    if(lat_):
        coor_lat = lat_
    if(long_):
        coor_long = long_
    return jsonify({'message':'done'})


@app.route('/detect', methods=['GET'])
def detect():
    if(exit_detect_event.is_set()):
        id = request.args.get('id', default = None)
        save_img = request.args.get('save_img', default = False)
        if id and id in all_trips:
            return jsonify({'message':'done'})
        else: return return_error('The trip does not exist', 404)
    else: return return_error('There is a detection already', 404)

@app.route('/stop_detect', methods=['GET'])
def stop_detect():
    if(not exit_detect_event.is_set()):
        exit_detect_event.set()
        return jsonify({'message':'done'})
    else:
        return return_error('no detection executing', 404) 

@app.route('/trips', methods=['GET'])
def list_trips():
    results = []
    for key in all_trips:
        results.append(all_trips[key])
    return jsonify(results)


@app.route('/trip/create', methods=['PUT'])
def create_trip():
    global all_trips
    request_json = request.get_json()
    name = request_json.get("name", None)
    if name:
        trips_management.create_trip(TRIPS_FOLDER, name)
        all_trips = trips_management.read_all_trips(TRIPS_FOLDER)
        return jsonify(
                {
                    'message' : 'Trip created'
                })
    return return_error('You have to provide a name', 404)

@app.route('/trip/delete', methods=['PUT'])
def delete_trip():
    global all_trips
    request_json = request.get_json()
    id = request_json.get("id", None)
    if id:
        if trips_management.delete_trip(TRIPS_FOLDER, id):
            all_trips = trips_management.read_all_trips(TRIPS_FOLDER)
            return jsonify(
                {
                    'message' : 'Trip deleted'
                })
        else:
            return return_error('The trip does not exist', 404)
    return return_error('You have to provide an id', 404)

@app.route('/trip/stats', methods=['GET'])
def trip_stats():
    id = request.args.get('id', default = None)
    if id:
        stats = trips_management.get_stats(TRIPS_FOLDER, id)
        if stats:
            return jsonify(
                {
                    'data' : stats
                })
        else:
            return return_error('The trip does not exist', 404)
    return return_error('You have to provide an id', 404)

@app.route('/trip/image', methods=['GET'])
def get_image():
    id = request.args.get('id', default = None)
    detection_number = request.args.get('detection_number', default = None)
    if id and detection_number:
        if id in all_trips:
            image = trips_management.get_image(IMAGES_FOLDER, id, detection_number, app)
            if image:
                return image
            else:
                return return_error('The specified image does not exist', 404)
        else:
            return return_error('The trip does not exist', 404)
    return return_error('You have to provide an id and a number', 404)

if __name__ == '__main__':
    app.debug = True
    app.run(host= '0.0.0.0')