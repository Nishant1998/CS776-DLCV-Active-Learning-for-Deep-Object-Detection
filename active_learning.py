from utils.data_handlers import *
from utils.training_functions import *
from utils.mAP_function import getDataForMap

## Setting data
## making annotation files
make_annotation_txtfile()
select_train_data(100) # select  init data for training

BATCH_SIZE = 16
EPOCHS = 5
log_dir = 'logs/000/'

# train intial model
train(batch=BATCH_SIZE, epoch=EPOCHS, load_pretrained=True, weights_path="./yolocode/model_data/yolo_weights.h5")
AL_map = []

# active learning loop
for i in range(100):
    # select new images for training
    print('Active Learning Iteration ', i)
    ## change score_type to select scoring function normal / prob (for uncertanty based scoring function)
    ## annotate image based on previously trained model
    annotated_images(model_path="./logs/000/trained_weights.h5", n=30, score_type='normal')

    # train new model with new training images selected
    train(batch=BATCH_SIZE, epoch=EPOCHS, load_pretrained=True, weights_path="./yolocode/model_data/yolo_weights.h5")
    shutil.copyfile(log_dir + "/trained_weights.h5", log_dir + "/trained_weights_" + str(i) + ".h5")

    # calculate mAP score
    preds, target = getDataForMap(log_dir + "/trained_weights_" + str(i) + ".h5")
    metric = MeanAveragePrecision.MAP()
    metric.update(preds, target)
    pprint(metric.compute())
    AL_map.append(metric.compute())