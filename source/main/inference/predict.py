import torch
import joblib
import pandas as pd
from .detect import config_dict
from .model import prepare_model
from .calculation import *

def predict_anomaly(feature, model_path, device):
    print("start inference")
    #yes가 1임

    model = prepare_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    feature = feature.unsqueeze(0)
    expanded_data = torch.stack([feature]*3, dim=1)
    expanded_data = expanded_data.to(device)
    output = model(expanded_data)
    pred = get_likely_index(output)

    print( output )
    print( "이상치 감지" if int(pred[0]) == 1 else "이상치 불검출")
    horn = True if int(pred[0]) == 1 else False
    return horn

def predict_location(peak, average_velocity):
    data = pd.DataFrame({"peak": [peak], "average_velocity(km)":[average_velocity]})
    regressor = joblib.load(config_dict['regressor'])
    prediction = regressor.predict(data)
    print( "위치 추정: ", str(prediction[0]) + "m" )
