import torch
import joblib
import pandas as pd
from .detect import config_dict
from .model import prepare_model
from .calculation import *
import copy
import xgboost
class Ensemble():
  def __init__(self):
    dir = "source/result/trained_model/"
    self.model_path_list = [dir+"best_model_fold0.pt", dir+"best_model_fold1.pt", dir+"best_model_fold2.pt"]
    self.model_list = []
    for path in self.model_path_list:
        model = prepare_model()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.to('cpu')

        self.model_list.append(copy.deepcopy(model))
        model = None

  def __call__(self,x):
    tensor_list = []
    for model in self.model_list:
        model.eval()
        output = model(x)
        tensor_list.append( get_likely_index(output) )

    prediction_list = torch.stack(tensor_list, dim=0)
    ensemble = torch.sum(prediction_list, 0)
    ensemble_output = torch.where(ensemble>1, 1, 0)

    print(ensemble_output)
    return ensemble_output
  
def predict_anomaly(feature, model_path, device, mode="lightweight"):
    print("<<<---start inference--->>>")
    #yes가 1임

    if mode == "lightweight":
        print("Single Model")
        model = prepare_model()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to("cpu")
        model.eval()
    elif mode == "high_accuracy":
        print("Ensemble Model")
        model = Ensemble()

    feature = feature.unsqueeze(0)
    expanded_data = torch.stack([feature]*3, dim=1)
    expanded_data = expanded_data.to("cpu")
    output = model(expanded_data)
    if mode == "lightweight":
        output = get_likely_index(output)

    print( "이상치 감지" if int(output[0]) == 1 else "이상치 불검출")
    horn = True if int(output[0]) == 1 else False
    return horn

def predict_location(peak, average_velocity):
    data = pd.DataFrame({"peak": [peak], "average_velocity(km)":[average_velocity]})

    try:
        regressor = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=5)
        regressor.load_model(config_dict['regressor'])

        # regressor_2 = xgboost.XGBRegressor(n_estimators=30, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=5)
        # regressor_2.load_model("source/result/trained_model/xgb_regressor.model")

        linear_regressor = joblib.load("source/result/trained_model/linear_regression.pkl")

        prediction_1 = regressor.predict(data.values)
        prediction_2 = linear_regressor.predict(data.values)
        # prediction_3 = regressor.predict(data.values)

    except Exception as e:
        print(e)
    print(data, prediction_1, prediction_2)
    prediction = prediction_1[0]
    print( "위치 추정: ", str(prediction) + "m" )
