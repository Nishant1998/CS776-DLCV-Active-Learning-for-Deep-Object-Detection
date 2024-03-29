import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from ..yolocode.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from ..yolocode.yolo3.utils import get_random_data


# 1
def get_classes(classes_path="./yolocode/model_data/coco_classes.txt"):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


# 2
def get_anchors(anchors_path="./yolocode/model_data/yolo_anchors.txt"):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


# 3
def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=0,
                 weights_path='./yolocode/model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for l in range(3)]
    ## create list of o/p shape -
    ## shape=(None, 13, 13, num_anchors/3 , num_classes+5);
    ## shape=(None, 26, 26, num_anchors/3, num_classes+5);
    ## shape=(None, 52, 52, num_anchors/3, num_classes+5)
    ## +5 for [xmin,ymin,xmax,ymax,class] for all classes and all anchore div into 3
    # load model architecture
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


# 4
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


# 5
def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


# 6
def train(batch=32, epoch=50, load_pretrained=False, weights_path="./logs/000/trained_weights.h5"):
    annotation_path = './coco_train.txt'
    log_dir = 'logs/000/'
    classes_path = './yolocode/model_data/coco_classes.txt'
    anchors_path = './yolocode/model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)  ## get list of names
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)  ## get list of 2d lists

    input_shape = (416, 416)  # multiple of 32, hw

    ## LOAD MODEL
    model = create_model(input_shape, anchors, num_classes, load_pretrained, weights_path)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss',
                                 save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # read train annotaion and split 90% train 10% val
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        # print('Unfreeze all of the layers.')
        batch_size = batch  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=epoch,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights.h5')

    return model